import argparse
import copy
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .model import VideoModel, adapt, image_tensor, prequential


def prepare(args):
    manifest = Path(args.manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups, records = {}, []
    for number, row in enumerate(rows):
        role = row["role"]
        if role not in ("train", "val", "stream"):
            raise ValueError("Roles are train, val, stream")
        group = row["group"]
        if group in groups and groups[group] != role:
            raise ValueError("A physical video group cannot cross data roles")
        groups[group] = role
        source = manifest.parent / row["path"]
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {source}")
        frames, index = [], 0
        while True:
            successful, frame = capture.read()
            if not successful:
                break
            if index % args.stride == 0:
                frames.append(cv2.cvtColor(cv2.resize(frame, (args.size, args.size)), cv2.COLOR_BGR2RGB))
            index += 1
        capture.release()
        if len(frames) < 2:
            raise ValueError(f"At least two decoded frames required: {source}")
        name = f"clip_{number:06d}.npz"
        np.savez_compressed(output / name, frames=np.asarray(frames, dtype=np.uint8))
        records.append(dict(file=name, label=int(row["label"]), role=role, group=group,
                            order=int(row["order"]), boundary=bool(int(row.get("boundary", 0))),
                            stream=row.get("stream", "main")))
    if {record["role"] for record in records} != {"train", "val", "stream"}:
        raise ValueError("All three roles must be present")
    labels = sorted(set(record["label"] for record in records))
    if labels != list(range(max(labels) + 1)):
        raise ValueError("Consecutive class IDs starting at zero required")
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_frames(root, record):
    with np.load(root / record["file"], allow_pickle=False) as archive:
        return archive["frames"]


def train(args):
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    classes = max(record["label"] for record in records) + 1
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = VideoModel(classes, args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    training = [record for record in records if record["role"] == "train"]
    validation = [record for record in records if record["role"] == "val"]
    history, best = [], -1
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(training)):
            record = training[index]
            frames = load_frames(root, record)
            indices = np.sort(rng.choice(len(frames), min(args.frames, len(frames)), replace=False))
            logits, features, _ = model(image_tensor(frames[indices], args.device))
            hidden = (torch.rand((len(indices), 1, *features.shape[-2:]), device=args.device) < 0.25).float()
            recovered = model.predictor(features * (1 - hidden))
            reconstruction = ((recovered - features.detach()).square() * hidden).sum() / (hidden.sum() * features.shape[1]).clamp_min(1)
            objective = F.cross_entropy(logits[None], torch.tensor([record["label"]], device=args.device)) + args.mask_weight * reconstruction
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach()))
        model.eval()
        correct = []
        with torch.no_grad():
            for record in validation:
                frames = load_frames(root, record)
                indices = np.linspace(0, len(frames) - 1, min(args.frames, len(frames))).astype(int)
                logits, _, _ = model(image_tensor(frames[indices], args.device))
                correct.append(int(logits.argmax()) == record["label"])
        accuracy = float(np.mean(correct))
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_accuracy=accuracy))
        if accuracy > best:
            best = accuracy
            torch.save(dict(state=model.state_dict(), classes=classes, width=args.width, config=vars(args)), output / "source.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")


@torch.no_grad()
def loss_value(model, previous, current, args, index):
    return float(prequential(model, previous, current, args, index, args.method == "entropy_only", args.method == "no_motion"))


def recovery(target, predicted, boundaries, args):
    correct = np.asarray(target) == predicted
    records = []
    for boundary in boundaries:
        if boundary < args.recovery_window:
            continue
        band = float(correct[boundary - args.recovery_window:boundary].mean() - args.recovery_tolerance)
        recovered = None
        for end in range(boundary + args.recovery_window, len(correct) + 1):
            if correct[end - args.recovery_window:end].mean() >= band:
                recovered = end - boundary
                break
        records.append(dict(boundary_frame=boundary, target_accuracy=band, recovery_frames=recovered))
    return records


def run(args):
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source, map_location=args.device, weights_only=True)
    source = VideoModel(checkpoint["classes"], checkpoint["width"]).to(args.device)
    source.load_state_dict(checkpoint["state"])
    source.eval()
    records = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = sorted([row for row in records if row["role"] == "stream" and row["stream"] == args.stream], key=lambda row: row["order"])
    if not records:
        raise ValueError("No clips found for this stream")
    deployed, anchor, candidate = copy.deepcopy(source), copy.deepcopy(source), None
    proposal, evidence, rollback_score, last_proposal = -1, 0.0, 0.0, -args.proposal_interval
    upper, lower = np.log((1 - args.beta) / args.alpha), np.log(args.beta / (1 - args.alpha))
    target, prediction, source_losses, live_losses, rows, events, boundaries = [], [], [], [], [], [], []
    audits, completed, advantages = [], [], []
    index = 0
    started = time.perf_counter()
    for record in records:
        frames = load_frames(root, record)
        buffer = deque(maxlen=args.window)
        previous = None
        if record["boundary"]:
            boundaries.append(index)
        for current in frames:
            buffer.append(current)
            with torch.no_grad():
                live, _, _ = deployed(image_tensor(list(buffer), args.device))
                reference, _, _ = source(image_tensor(list(buffer), args.device))
                label = torch.tensor([record["label"]], device=args.device)
                live_losses.append(float(F.cross_entropy(live[None], label)))
                source_losses.append(float(F.cross_entropy(reference[None], label)))
                prediction.append(int(live.argmax()))
            target.append(record["label"])
            if previous is not None:
                deployed_loss = loss_value(deployed, previous, current, args, index)
                anchor_loss = loss_value(anchor, previous, current, args, index)
                rollback_score = max(0, rollback_score + deployed_loss - anchor_loss - args.rollback_drift)
                for audit in audits:
                    if index <= audit["frame"] + 1:
                        continue
                    audit["difference"].append(loss_value(audit["after"], previous, current, args, index)
                                               - loss_value(audit["before"], previous, current, args, index))
                remaining = []
                for audit in audits:
                    if len(audit["difference"]) >= args.audit_frames:
                        completed.append(dict(commit_frame=audit["frame"], mean_prequential_change=float(np.mean(audit["difference"])),
                                              harmful=bool(np.mean(audit["difference"]) > args.harm_tolerance)))
                    else:
                        remaining.append(audit)
                audits = remaining
                if candidate is not None and index > proposal + 1:
                    candidate_loss = loss_value(candidate, previous, current, args, index)
                    advantage = deployed_loss - candidate_loss
                    evidence += ((advantage - args.null_mean) ** 2 - (advantage - args.alternative_mean) ** 2) / (2 * args.evidence_std ** 2)
                    advantages.append(advantage)
                    commit = evidence >= upper
                    reject = evidence <= lower
                    if args.method == "fixed_window":
                        commit = len(advantages) >= args.decision_window and np.mean(np.asarray(advantages) > 0) > 0.5
                        reject = len(advantages) >= args.decision_window and not commit
                    if commit:
                        before = copy.deepcopy(deployed)
                        anchor, deployed = before, candidate
                        audits.append(dict(frame=index, before=copy.deepcopy(before), after=copy.deepcopy(deployed), difference=[]))
                        events.append(dict(frame=index, action="commit", proposal=proposal, delay=index - proposal, evidence=evidence))
                        candidate, rollback_score = None, 0.0
                    elif reject:
                        events.append(dict(frame=index, action="reject", proposal=proposal, evidence=evidence))
                        candidate = None
                if args.method not in ("source", "no_rollback", "tent", "immediate") and rollback_score > args.rollback_threshold:
                    deployed = copy.deepcopy(anchor)
                    candidate, rollback_score = None, 0.0
                    events.append(dict(frame=index, action="rollback"))
            if args.method != "source" and candidate is None and index - last_proposal >= args.proposal_interval and len(buffer) >= 2:
                candidate = adapt(copy.deepcopy(deployed), list(buffer), args, index)
                proposal, last_proposal, evidence, advantages = index, index, 0.0, []
                if args.method in ("immediate", "tent"):
                    before = copy.deepcopy(deployed)
                    anchor, deployed = before, candidate
                    audits.append(dict(frame=index, before=before, after=copy.deepcopy(deployed), difference=[]))
                    events.append(dict(frame=index, action="commit", proposal=index, delay=0, evidence=None))
                    candidate, rollback_score = None, 0.0
            rows.append(dict(frame=index, label=record["label"], prediction=prediction[-1], pending=candidate is not None,
                             evidence=evidence if candidate is not None else None, rollback_score=rollback_score))
            previous = current
            index += 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    delays = [event["delay"] for event in events if event["action"] == "commit"]
    result = dict(config=vars(args), online_accuracy=float(np.mean(np.asarray(target) == prediction)),
                  cumulative_regret=float(np.sum(np.asarray(live_losses) - source_losses)),
                  harmful_update_rate=float(np.mean([row["harmful"] for row in completed])) if completed else None,
                  audited_commits=len(completed), unresolved_commit_audits=len(audits), commits=len(delays),
                  mean_commit_delay=float(np.mean(delays)) if delays else None, fps=index / elapsed,
                  recovery=[] if args.method == "source" else recovery(target, prediction, boundaries, args),
                  pending_candidate=candidate is not None)
    for name, value in (("metrics", result), ("events", events), ("stream", rows), ("commit_audits", completed)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    torch.save(dict(state=deployed.state_dict(), classes=checkpoint["classes"], width=checkpoint["width"]), output / "deployed.pt")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--manifest", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--size", type=int, default=128)
    data.add_argument("--stride", type=int, default=1)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    for name, default in (("width", 64), ("frames", 8), ("epochs", 50), ("seed", 13)):
        training.add_argument(f"--{name}", type=int, default=default)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--mask-weight", type=float, default=0.1)
    training.add_argument("--device", default="cpu")
    execution = sub.add_parser("run")
    execution.add_argument("--data", required=True)
    execution.add_argument("--source", required=True)
    execution.add_argument("--output", required=True)
    execution.add_argument("--stream", default="main")
    execution.add_argument("--method", choices=["full", "source", "tent", "immediate", "entropy_only", "no_motion", "fixed_window", "no_rollback"], default="full")
    for name, default in (("window", 16), ("proposal-interval", 16), ("adapt-steps", 3), ("audit-frames", 30),
                          ("decision-window", 20), ("recovery-window", 50), ("seed", 13)):
        execution.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("adapt-lr", 0.0001), ("alpha", 0.05), ("beta", 0.1), ("null-mean", 0),
                          ("alternative-mean", 0.01), ("evidence-std", 0.1), ("rollback-threshold", 1),
                          ("rollback-drift", 0.001), ("motion-weight", 1), ("feature-weight", 0.1),
                          ("track-weight", 0.1), ("harm-tolerance", 0), ("recovery-tolerance", 0.05)):
        execution.add_argument(f"--{name}", type=float, default=default)
    execution.add_argument("--device", default="cpu")
    plot = sub.add_parser("plot")
    plot.add_argument("--runs", nargs="+", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if min(args.stride, args.size) < 1:
            parser.error("Positive size and stride required")
        prepare(args)
    elif args.command == "train":
        if args.width < 2 or min(args.frames, args.epochs, args.lr) <= 0:
            parser.error("Positive training sizes and width at least two required")
        train(args)
    elif args.command == "run":
        if not 0 < args.alpha < 1 or not 0 < args.beta < 1 or min(args.evidence_std, args.proposal_interval, args.audit_frames, args.adapt_steps) <= 0:
            parser.error("Invalid sequential-test or adaptation parameters")
        run(args)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for metric in ("online_accuracy", "cumulative_regret", "harmful_update_rate", "mean_commit_delay", "fps"):
            records = [json.loads((Path(run) / "metrics.json").read_text(encoding="utf-8")) for run in args.runs]
            records = [record for record in records if record[metric] is not None]
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.bar([row["config"]["method"] for row in records], [row[metric] for row in records])
            axis.set_ylabel(metric)
            axis.tick_params(axis="x", rotation=25)
            figure.tight_layout()
            figure.savefig(output / f"{metric}.png", dpi=180)
            plt.close(figure)
