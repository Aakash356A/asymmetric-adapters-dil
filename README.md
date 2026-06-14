# DCASE 2026 Task 7 – Domain-Agnostic Incremental Learning for Audio Classification

Experiments for DCASE 2026 Task 7. The submitted system (`bn_res_deep_clf`) extends the baseline with per-domain residual bottleneck adapters of asymmetric depth (D2=1, D3=3), a warm-started per-domain classifier head, and a supervised contrastive auxiliary loss, on top of a shared frozen CNN14 backbone with per-domain BatchNorm.

**Submitted results (development set, domain-agnostic ADIL inference):**

| Step | D2 | D3 | Avg |
|---|---|---|---|
| After D2 training | 76.21% | — | — |
| After D3 training | 76.21% | 60.79% | 68.5% |

---

## Model Architecture

![Model Architecture](Dcase_task7_architecture.png)

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

Pass `--resume` with the same arguments used during training. The script loads the saved checkpoint, skips training, and prints per-domain and per-class accuracy:

```bash
python code/baseline/baseline_DIL_task7.py train \
  --model_type bn_res_deep_clf \
  --adapter_depths 1,3 \
  --lambda_con 0.2 \
  --epoch 180 \
  --checkpoint_dir checkpoints/BN_RES_DEEP_CLF_lr1e-3_bs32_ep180_deep_clf_d3_ep180 \
  --resume \
  --learning_rate 1e-3 --batch_size 32 --num_workers 4 --cuda --augmentation none
```

Output includes overall accuracy and a per-class breakdown for each domain:

```
seen domain: ['D2'] and its accuracy: 77.93
    alarm: 77.7%
    bark: 95.7%
    ...
seen domain: ['D3'] and its accuracy: 58.81
    alarm: 74.2%
    baby_cry: 50.0%
    ...
```

### Option D – Verify the submission model definition

The submission folder contains a standalone model file (`ResAdapt.py`) and a self-check script that confirms it loads the provided checkpoints correctly, without needing the full training codebase:

```bash
cd submission/Divakar_IND_task7_1
python verify_submission.py
```

This will:
1. Import `ResAdapt.py` and build the model with the submitted adapter depths
2. Load `checkpoint/checkpoint_D2.pth` and `checkpoint/checkpoint_D3.pth` with strict key matching
3. Run a forward pass for every domain task on a dummy 4 s / 32 kHz waveform
4. Run the domain-agnostic (ADIL) entropy-routing inference

Expected output:
```
[D2] loading checkpoint/checkpoint_D2.pth
  OK — exact match (strict load would also succeed)
[D3] loading checkpoint/checkpoint_D3.pth
  OK — exact match (strict load would also succeed)
forward pass on dummy input (2, 128000):
  task 0: logits (2, 10)
  task 1: logits (2, 10)
  task 2: logits (2, 10)
  ADIL domain-agnostic prediction: [3, 7]
ALL CHECKS PASSED — .py and .pth files are self-consistent.
```

Exit code `0` means the submission `.py` and `.pth` files are self-consistent and ready to submit.

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

**Reproduced baseline (BN only):** D2 ≈ 59.2%, D3 ≈ 40.8%, Avg ≈ 50.0%

**Submitted system (bn_res_deep_clf, 180 epochs):** D2 = 76.21%, D3 = 60.79%, Avg = 68.5%

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
| `--adapter_depths` | — | Per-task ResBottleneck depth, comma-sep (e.g. `1,3` → D2=1, D3=3) |
| `--lambda_con` | `0.0` | Supervised contrastive loss weight (0 = off, submitted = 0.2) |
| `--tta_n` | `1` | Test-time augmentation shifts to average (1 = off) |
| `--unfreeze_blocks` | `0` | Unfreeze top N CNN14 conv blocks for all domains |
| `--unfreeze_blocks_per_task` | — | Per-task unfreeze, comma-sep (e.g. `0,2` = D2 frozen, D3 unfreezes 2) |
| `--lambda_l2_backbone` | `0.0` | L2 anchor penalty toward previous-domain backbone weights |

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
