"""Train one model from the A-F ablation family."""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/F_yolo11s_DWR_EIEStem_CARAFE_P2.yaml")
    parser.add_argument("--data", required=True, help="Path to an Ultralytics dataset YAML file.")
    parser.add_argument("--weights", default=None, help="Optional checkpoint used to initialize compatible layers.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model, task="detect")
    if args.weights:
        model.load(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
