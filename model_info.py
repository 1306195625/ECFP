"""Build all A-F YAML files and print model summaries."""

from pathlib import Path

from ultralytics import YOLO


def main():
    for config in sorted(Path("models").glob("[A-F]_*.yaml")):
        print(f"\n{'=' * 20} {config.name} {'=' * 20}")
        YOLO(config, task="detect").info(detailed=False, verbose=True)


if __name__ == "__main__":
    main()
