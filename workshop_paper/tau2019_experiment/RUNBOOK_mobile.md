# TAU Mobile DIL experiment — runbook (device-as-domain)

Alternative / complementary benchmark to the cities run: use **recording
devices** as the incremental domains (D1=device A → D2=B → D3=C). Same 10
scene classes, so it reuses `config_task7.py` (tau mode) and `train_d1_base.py`
**unchanged** — only a different prep script and a separate data/checkpoint
folder. The cities scripts are not touched.

> Use **TAU2020 Mobile**, not 2019 Mobile. Honest reason: 2019 Mobile's B/C
> devices have only ~18 min each — too little to train per-device adapters on.
> 2020 Mobile has adequate B/C data *and* simulated devices S1–S6 with a
> built-in unseen-device split (S4–S6 are test-only), which is exactly what
> makes the routing/rejection story rigorous.

Downloads: [TAU2020 Mobile dev](https://zenodo.org/records/3670167) (**27.4 GB**,
a bit smaller than the 35.6 GB cities set) + the same PANNs `Cnn14_mAP=0.431.pth`.

`$REPO` = the `dcase_task7` directory. Note the **separate** data dir
(`tau_mobile_data`) and checkpoint dir (`TAU_mobile_run`) so this never
collides with the cities run.

---

## 1. Build the device-as-domain dataset

```bash
python workshop_paper/tau2019_experiment/prepare_tau_mobile.py \
  --tau_root /path/to/TAU-urban-acoustic-scenes-2020-mobile-development \
  --devices a,b,c \
  --unseen_devices s4,s5,s6 \
  --out $REPO/tau_mobile_data
```

- D1=device A, D2=device B, D3=device C (official train/test split per file).
- `--unseen_devices s4,s5,s6` writes `tau_mobile_data/evaluation_setup/unseen_test.txt`
  — held out, **not** trained on — for the unseen-device routing analysis.
- Fast smoke test first: add `--limit_per_device 80`.

---

## 2. Train the D1 base (device A) → `checkpoint_D1.pth`

Reuses the cities trainer; just point `TAU_DATA_DIR` at the mobile folder and
use a separate checkpoint dir.

```bash
TAU_DATA_DIR=$REPO/tau_mobile_data \
python workshop_paper/tau2019_experiment/train_d1_base.py \
  --checkpoint_dir $REPO/checkpoints/TAU_mobile_run \
  --panns_ckpt /path/to/Cnn14_mAP=0.431.pth \
  --epochs 80 --batch_size 32 --lr 1e-3 --num_workers 8 --cuda
```

Device A is the large partition (~40 h), so this base is data-rich — good, but
**not** faster than the cities D1. The speed win is only on D2/D3 (small).

---

## 3. Incremental training on D2, D3 (devices B, C)

```bash
DCASE_DATASET=tau TAU_DATA_DIR=$REPO/tau_mobile_data \
python code/baseline/baseline_DIL_task7.py train \
  --model_type bn_res_deep_clf \
  --adapter_depths 1,3 \
  --lambda_con 0.2 \
  --epoch 180 \
  --checkpoint_dir $REPO/checkpoints/TAU_mobile_run \
  --learning_rate 1e-3 --batch_size 32 --num_workers 8 \
  --cuda --augmentation none --save --class_balance
```

- `--class_balance` is recommended here because device partitions are uneven.
- Baseline row for the table: rerun with `--model_type bn` into a separate
  `--checkpoint_dir`.

---

## 4. (Optional but high value) Unseen-device routing analysis

The `unseen_test.txt` clips (S4–S6) are the on-benchmark analogue of the
ESC-50 probe: same task/labels, genuinely unseen devices. Run the trained
model's entropy router over them and report accuracy + mean entropy + routed
D1/D2/D3 fractions — the same three columns as Table `tab:esc50`. If entropy
routing is again over-confident on unseen devices, that turns the ESC-50
side-note into a rigorous, standard-benchmark result. (Ask me to adapt
`workshop_paper/esc50_experiment/run_esc50_probe.py` for this once the mobile
checkpoints exist — it's a small change to point at `unseen_test.txt` and the
mobile checkpoint.)

---

## Notes

- **Separate `--checkpoint_dir` and `TAU_DATA_DIR` from the cities run.** Both
  use `DCASE_DATASET=tau`; the data folder is what distinguishes them.
- Same `--checkpoint_dir` for steps 2 and 3 (that's how D1 is handed off).
- Task7 workflow is unaffected (env switch only activates with `DCASE_DATASET=tau`).
- Is Mobile "smaller/quicker"? Download yes (27.4 vs 35.6 GB); training only
  partly (D1/device-A is still big). The real reason to prefer it is the
  **unseen-device split**, not speed.
