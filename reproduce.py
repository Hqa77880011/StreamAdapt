"""Rendered geometric-object videos with source/validation/stream groups; CPU CNN32, seed13 and future proxy gate."""

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
    source = args.manifest or str(output / 'generated/videos.csv')
    steps = []
    if not args.manifest:
        steps.append(['finite_data.py', '--output', str(output / 'generated'), '--seed', '13'])
    return steps + [
        ["-m", "streamadapt", "prepare", "--manifest", source, "--output", str(output / "data"), "--size", "128", "--stride", "1"],
        ["-m", "streamadapt", "train", "--data", str(output / "data"), "--output", str(output / "source"), "--width", "32", "--frames", "8", "--epochs", "50", "--seed", "13"],
        ["-m", "streamadapt", "run", "--data", str(output / "data"), "--source", str(output / "source/source.pt"), "--stream", "main", "--method", "full", "--seed", "13", "--window", "16", "--audit-frames", "30", "--task-harm-tolerance", "0", "--output", str(output / "full")],
        ["-m", "streamadapt", "run", "--data", str(output / "data"), "--source", str(output / "source/source.pt"), "--stream", "main", "--method", "source", "--seed", "13", "--window", "16", "--audit-frames", "30", "--task-harm-tolerance", "0", "--output", str(output / "source")],
        ["-m", "streamadapt", "run", "--data", str(output / "data"), "--source", str(output / "source/source.pt"), "--stream", "main", "--method", "immediate", "--seed", "13", "--window", "16", "--audit-frames", "30", "--task-harm-tolerance", "0", "--output", str(output / "immediate")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/finite-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--manifest', type=absolute_path, help='Optional prepared input; otherwise generate the finite setting')
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
