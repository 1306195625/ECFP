"""Train all six ablation configurations sequentially."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/ablation")
    args = parser.parse_args()

    for config in sorted(Path("models").glob("[A-F]_*.yaml")):
        YOLO(config, task="detect").train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=config.stem,
        )


if __name__ == "__main__":
    main()
