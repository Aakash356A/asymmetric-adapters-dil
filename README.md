# DCASE 2026 Task 7 – Domain-Agnostic Incremental Learning for Audio Classification

Experiments for DCASE 2026 Task 7. The baseline uses domain-specific BatchNorm (BN) layers adapted incrementally across domains D1 → D2 → D3.

---

## Setup

### 1. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 2. Download the dataset

Download the development dataset from [Zenodo](https://zenodo.org/records/19335184) and arrange it as:

```
task7_data/
├── audio/
└── evaluation_setup/
    ├── development_train.txt
    └── development_test.txt
```

### 3. Download pretrained checkpoints

Download `checkpoint_D1.pth`, `checkpoint_D2.pth`, and `checkpoint_D3.pth` from [Google Drive](https://drive.google.com/drive/folders/1C21s1KlN4ZbnboIdtO8H0T-S0_k8yw3o?usp=sharing) and place them in `pretrained_models/`.

```
pretrained_models/
├── checkpoint_D1.pth
├── checkpoint_D2.pth
└── checkpoint_D3.pth
```

---

## Running experiments

### Option A – Run all experiments with the shell script (recommended)

```bash
bash run_experiments.sh
```

To also save a full log:

```bash
bash run_experiments.sh 2>&1 | tee runner.log
```

Results are appended to `experiment_results.txt` after each run. Each experiment gets its own checkpoint directory under `checkpoints/<experiment_name>/` and a per-run `train.log`.

Before running, update `BASE_DIR` at the top of `run_experiments.sh` to match your local path:

```bash
BASE_DIR="/path/to/dcase_task7"
```

### Option B – Run a single training experiment manually

```bash
python baseline/baseline_DIL_task7.py train \
  --augmentation=none \
  --learning_rate=1e-4 \
  --batch_size=32 \
  --cuda \
  --num_workers=16 \
  --epoch=120 \
  --model_type=bn \
  --checkpoint_dir=checkpoints/my_run \
  --save
```

### Option C – Evaluate using pre-trained checkpoints (skip training)

```bash
python baseline/baseline_DIL_task7.py train \
  --augmentation=none \
  --learning_rate=1e-4 \
  --batch_size=32 \
  --cuda \
  --num_workers=16 \
  --epoch=120 \
  --resume
```

---

## Experiment configurations in `run_experiments.sh`

| Tag | `--model_type` | Description |
|---|---|---|
| `baseline_repro3` | `bn` | Standard BN baseline (reference) |
| `perdomain_head` | `bn_clf` | Per-domain linear classification head |
| `perdomain_cosine` | `bn_cos` | Per-domain cosine / NCM head |
| `film_adapters` | `bn_film` | Per-domain FiLM adapters |
| `se_adapters` | `bn_se` | Per-domain Squeeze-and-Excitation adapters |
| `attention_pooling` | `bn_att` | Per-domain attention pooling |

All experiments use `lr=1e-4`, `bs=32`, `epochs=120` so architecture is the only variable.

To add a new experiment, append a line to the experiments section of `run_experiments.sh`:

```bash
#          model_type  lr      bs    epochs   [tag]
run_experiment bn_clf   5e-4    32    120      perdomain_head_lr5e4
```

---

## Reading results

```bash
cat experiment_results.txt
```

Each line shows: `timestamp | experiment_name | D2 accuracy | D3 accuracy | average accuracy | status`.

**Baseline target:** D2 ≈ 54.7%, D3 ≈ 35.0%, Avg ≈ 44.8%

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--augmentation` | required | `none` or `mixup` |
| `--learning_rate` | required | e.g. `1e-4` |
| `--batch_size` | required | e.g. `32` |
| `--epoch` | required | e.g. `120` |
| `--model_type` | — | Architecture variant (see table above) |
| `--checkpoint_dir` | — | Where to save checkpoints |
| `--cuda` | off | Enable GPU |
| `--num_workers` | required | DataLoader workers |
| `--save` | off | Save checkpoints during training |
| `--resume` | off | Load existing checkpoint and evaluate |

---

## Citation

```bibtex
@INPROCEEDINGS{10890481,
  author={Manjunath Mulimani and Annamaria Mesaros},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Domain-Incremental Learning for Audio Classification},
  year={2025},
  pages={1-5}
}
```
