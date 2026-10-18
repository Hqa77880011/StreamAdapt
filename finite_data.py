"""Render moving geometric objects into actual videos with independent source groups."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def render(shape, position, color, brightness, rng, size=128):
    """Pixels depend on physical object geometry, illumination and camera noise."""
    image = np.full((size, size, 3), 35 * brightness, np.float32)
    center = (np.asarray(position) * size).astype(int)
    radius = int(size * 0.15)
    ink = tuple(float(v * brightness) for v in color)
    if shape == "circle":
        cv2.circle(image, tuple(center), radius, ink, -1)
    elif shape == "square":
        cv2.rectangle(image, tuple(center - radius), tuple(center + radius), ink, -1)
    else:
        triangle = center + np.array([[0, -radius], [-radius, radius], [radius, radius]])
        cv2.fillPoly(image, [triangle], ink)
    return np.clip(image + rng.normal(0, 5, image.shape), 0, 255).astype(np.uint8)


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=False)
    (output / "clips").mkdir()
    rng, records, serial = np.random.default_rng(seed), [], 0
    shapes = ["circle", "square", "triangle"]
    for role, per_class in [("train", 6), ("val", 2), ("stream", 4)]:
        for repeat in range(per_class):
            for label, shape in enumerate(shapes):
                relative = f"clips/clip-{serial:04d}.avi"
                writer = cv2.VideoWriter(str(output / relative), cv2.VideoWriter_fourcc(*"MJPG"), 8, (128, 128))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV MJPG writer is unavailable")
                color = rng.uniform(120, 235, 3)
                phase = rng.uniform(-np.pi, np.pi)
                illumination = 1 if role != "stream" else (0.65 if repeat % 2 else 1.2)
                try:
                    for frame in range(24):
                        position = [0.5 + 0.22 * np.sin(frame / 10 + phase),
                                    0.5 + 0.18 * np.cos(frame / 12 + phase)]
                        rgb = render(shape, position, color, illumination, rng)
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                records.append(dict(path=relative, label=label, role=role, group=f"physical-clip-{serial}",
                                    stream="main", order=serial, boundary=int(role == "stream" and label == 0)))
                serial += 1
    with (output / "videos.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    generate(Path(args.output), args.seed)


if __name__ == "__main__":
    main()
