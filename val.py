"""Validate a trained checkpoint."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
