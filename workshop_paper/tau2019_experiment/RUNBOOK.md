# TAU2019 DIL experiment — runbook

Goal: validate the per-domain residual-adapter system on a **second, cleaner
benchmark** — TAU Urban Acoustic Scenes 2019, using three cities as the
incremental domains D1 → D2 → D3 (the domain shift is the change of city, as
in the ADIL paper). This reuses the *exact* model and training loop from
`code/baseline/baseline_DIL_task7.py`; only the dataset is swapped, via the
`DCASE_DATASET=tau` env switch added to `code/utils/config_task7.py`.

All commands are run from the repo root on your GPU box (RTX 6000 Ada).
`$REPO` = the `dcase_task7` directory.

---

## 0. One-time deps

The prep script writes wavs with `soundfile`; everything else you already
have. If needed:

```bash
pip install soundfile librosa tqdm
```

---

## 1. Downloads

**a) TAU Urban Acoustic Scenes 2019, development** (~35.6 GB) from Zenodo
record `2589280`. Download all `audio.*.zip` + `meta.zip` + `evaluation_setup`
and extract so you end up with one directory like:

```
TAU-urban-acoustic-scenes-2019-development/
├── audio/               # *.wav
├── meta.csv
└── evaluation_setup/
    ├── fold1_train.csv
    └── fold1_evaluate.csv
```

**b) PANNs Cnn14 checkpoint** (`Cnn14_mAP=0.431.pth`, ~300 MB) — the AudioSet
pretrained backbone, used to initialise D1. From Zenodo record `3987831`:

```bash
wget -O Cnn14_mAP=0.431.pth \
  "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
```

(You can skip PANNs and train the backbone from scratch, but PANNs init gives
a much stronger frozen backbone — recommended, since D2/D3 freeze it.)

---

## 2. Build the DIL dataset (3 cities → D1/D2/D3)

```bash
python workshop_paper/tau2019_experiment/prepare_tau2019.py \
  --tau_root /path/to/TAU-urban-acoustic-scenes-2019-development \
  --cities lisbon,lyon,prague \
  --out $REPO/tau_data
```

- Resamples to 32 kHz mono, writes `tau_data/D{1,2,3}-{train,test}/*.wav`, and
  `tau_data/evaluation_setup/development_{train,test}.txt` in the task7 schema.
- Uses the official `fold1_train` / `fold1_evaluate` split per file (no
  location leakage).
- `--cities` order = D1,D2,D3. Lisbon/Lyon/Prague mirrors the ADIL paper; any
  three TAU2019 dev cities work (barcelona, helsinki, london, lisbon, lyon,
  milan, paris, prague, stockholm, vienna). Each city ≈ 1440 clips / 10 scenes.

---

## 3. Train the D1 base model → `checkpoint_D1.pth`

```bash
python workshop_paper/tau2019_experiment/train_d1_base.py \
  --checkpoint_dir $REPO/checkpoints/TAU_run \
  --panns_ckpt /path/to/Cnn14_mAP=0.431.pth \
  --epochs 80 --batch_size 32 --lr 1e-3 --num_workers 8 --cuda
```

Trains the shared CNN14 backbone + D1 BN + D1 head on city-1, saving the best
model to `checkpoints/TAU_run/checkpoint_D1.pth` — exactly where the
incremental pipeline looks for it. (The script sets `DCASE_DATASET=tau`
itself.) Watch the printed `D1 test acc`; with PANNs init it should climb fast.

---

## 4. Incremental training on D2, D3 (your submitted config)

Same entrypoint you used for the challenge — just prefixed with the dataset
switch and pointed at the same checkpoint dir:

```bash
DCASE_DATASET=tau TAU_DATA_DIR=$REPO/tau_data \
python code/baseline/baseline_DIL_task7.py train \
  --model_type bn_res_deep_clf \
  --adapter_depths 1,3 \
  --lambda_con 0.2 \
  --epoch 180 \
  --checkpoint_dir $REPO/checkpoints/TAU_run \
  --learning_rate 1e-3 --batch_size 32 --num_workers 8 \
  --cuda --augmentation none --save
```

This loads `checkpoint_D1.pth`, trains D2 then D3 (freezing the backbone,
adding per-domain BN + residual adapters depth 1/3 + per-domain head), saves
`checkpoint_D2.pth` / `checkpoint_D3.pth`, and prints the domain-agnostic
(entropy-routed) accuracy for D2 and D3 plus the average.

**Baseline to compare against** — run the same command with
`--model_type bn` (plain per-domain BN, the ADIL baseline) into a *separate*
`--checkpoint_dir` to get the reference numbers for the ablation table.

---

## 5. Read the result

The run prints lines like `seen domain: ['D2'] and its accuracy: XX.X`. Those
D2/D3/avg numbers are what go into a new "TAU2019 (cross-city)" row alongside
the DCASE Task 7 numbers in the paper — showing the adapter system beats the
BN baseline on a *second, independent* benchmark, on cleaner audio.

---

## Notes / gotchas

- **Same `--checkpoint_dir` for steps 3 and 4** — that's how D1 is handed off.
- **Nothing about the task7 workflow changes.** With `DCASE_DATASET` unset the
  config behaves exactly as before; the switch only activates with `=tau`.
- Expected wall-clock on a single Ada 6000: D1 base ≈ 1–2 h (80 ep, PANNs
  init), D2+D3 ≈ 2–4 h (180 ep). Kick off D1 first; if its test acc looks
  healthy (well above chance = 10%), the incremental step is worth running.
- `prepare_tau2019.py --limit_per_city 60` gives a fast smoke test end-to-end
  before committing to the full run.
- The pipeline reports D2/D3 (matching how the paper reports the challenge).
  D1 is preserved bit-for-bit by construction (frozen backbone + untouched D1
  BN/head), so no separate forgetting check is required for the table.
