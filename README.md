# YOLO11-DWR-EIEStem-CARAFE-P2

This repository contains the complete **YOLO11s + DWR + EIEStem + CARAFE + P2** detector and the five models used in its ablation study. It is a focused, reproducible fork of Ultralytics 8.3.9.

## Model variants

| ID | DWR | EIEStem | CARAFE | P2 | Configuration |
|---|:---:|:---:|:---:|:---:|---|
| A |  |  |  |  | `models/A_yolo11s_baseline.yaml` |
| B | ✓ |  |  |  | `models/B_yolo11s_DWR.yaml` |
| C | ✓ | ✓ |  |  | `models/C_yolo11s_DWR_EIEStem.yaml` |
| D | ✓ | ✓ |  | ✓ | `models/D_yolo11s_DWR_EIEStem_P2.yaml` |
| E | ✓ | ✓ | ✓ |  | `models/E_yolo11s_DWR_EIEStem_CARAFE.yaml` |
| F | ✓ | ✓ | ✓ | ✓ | `models/F_yolo11s_DWR_EIEStem_CARAFE_P2.yaml` |

F is the complete proposed model. P2 denotes the additional high-resolution P2/4 detection branch rather than a standalone Python layer.

## Repository layout

```text
.
├── models/                         # A-F model YAML files
├── scripts/train_ablation.py       # sequential A-F training
├── ultralytics/                    # clean Ultralytics 8.3.9 source
│   └── nn/modules/ablation.py      # DWR, EIEStem and CARAFE
├── train.py
├── val.py
├── predict.py
├── model_info.py
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

The only custom neural-network implementation is in `ultralytics/nn/modules/ablation.py`. The corresponding YAML parser registration is in `ultralytics/nn/tasks.py`.

## Installation

Python 3.8 or newer is required.

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
python -m pip install -U pip
python -m pip install -e .
```

Install the CUDA build of PyTorch appropriate for your machine before the editable install when GPU training is required.

## Dataset

Use the standard Ultralytics detection layout and dataset YAML:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: class_0
  1: class_1
```

The `nc` field in the model YAML is automatically overridden by the number of names in the dataset YAML.

## Train

Train the complete model:

```bash
python train.py \
  --model models/F_yolo11s_DWR_EIEStem_CARAFE_P2.yaml \
  --data /path/to/data.yaml \
  --epochs 300 --imgsz 640 --batch 16 --device 0
```

Train a single ablation by changing `--model`, or train all A-F variants sequentially:

```bash
python scripts/train_ablation.py --data /path/to/data.yaml --device 0
```

## Validate and predict

```bash
python val.py --weights runs/train/exp/weights/best.pt --data /path/to/data.yaml
python predict.py --weights runs/train/exp/weights/best.pt --source /path/to/images --save
```

Build all six architectures without training:

```bash
python model_info.py
```

## Reproducibility notes

- The code base is pinned to the same upstream version as the experiments: Ultralytics 8.3.9.
- Training data, checkpoints, cache files and experiment outputs are intentionally excluded.
- The published F configuration uses `CARAFE, []`. Input channels are injected by the model parser; hard-coded channel arguments are incompatible with the included CARAFE constructor.
- Report software, CUDA, PyTorch, GPU, random seed, data split and training hyperparameters with any reproduced results.

## License and citation

This derivative work preserves the upstream **AGPL-3.0** license. Review the license obligations before redistribution or hosted deployment. `CITATION.cff` retains the upstream Ultralytics citation; add the associated paper citation before public release.
