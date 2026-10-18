"""From-scratch CPU CNN32 on 128px video, seed13, 50 epochs; past-window adaptation and future proxy gate, separate post-hoc task/accuracy harm."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "streamadapt", "prepare", "--manifest", args.manifest, "--output", str(output / "data"), "--size", "128", "--stride", "1"],
        ["-m", "streamadapt", "train", "--data", str(output / "data"), "--output", str(output / "source"), "--width", "32", "--frames", "8", "--epochs", "50", "--seed", "13"],
        ["-m", "streamadapt", "run", "--data", str(output / "data"), "--source", str(output / "source/source.pt"), "--stream", "main", "--method", "full", "--seed", "13", "--window", "16", "--audit-frames", "30", "--task-harm-tolerance", "0", "--output", str(output / "full")],
        ["-m", "streamadapt", "run", "--data", str(output / "data"), "--source", str(output / "source/source.pt"), "--stream", "main", "--method", "source", "--seed", "13", "--window", "16", "--audit-frames", "30", "--task-harm-tolerance", "0", "--output", str(output / "source")],
        ["-m", "streamadapt", "run", "--data", str(output / "data"), "--source", str(output / "source/source.pt"), "--stream", "main", "--method", "immediate", "--seed", "13", "--window", "16", "--audit-frames", "30", "--task-harm-tolerance", "0", "--output", str(output / "immediate")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--manifest', required=True, type=absolute_path)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
