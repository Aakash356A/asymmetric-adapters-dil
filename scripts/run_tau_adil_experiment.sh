#!/bin/bash
# ================================================================
# TAU2019 4-domain ADIL protocol runner
#
# Runs the full pipeline from the ADIL paper's 4-domain TAU2019 setup:
#   D1 = 6 pooled cities (barcelona+helsinki+london+paris+stockholm+vienna)
#   D2 = lisbon, D3 = lyon, D4 = prague
# with SYMMETRIC adapter depth (2,2,2) and class-balanced sampling.
#
# Usage:
#   bash scripts/run_tau_adil_experiment.sh
#   bash scripts/run_tau_adil_experiment.sh 2>&1 | tee runner.log
#
# Steps (each individually re-runnable — see SKIP_* flags below):
#   1. prepare_tau2019.py  — build the 4-domain manifest/wavs (shared across
#                            all experiments; only rebuild if it changes)
#   2. train_d1_base.py    — train D1 on the 6 pooled cities. Saved under a
#                            D1-only dir keyed by ADAPTER_DEPTHS, since the
#                            saved state_dict's shape depends on the adapter
#                            depths/nb_tasks the model was built with, so a
#                            D1 checkpoint is only reusable across runs that
#                            share the SAME depths.
#   3. baseline_DIL_task7.py train — incremental D2 -> D3 -> D4. Runs in its
#                            OWN per-experiment checkpoint dir (name encodes
#                            model/depths/epochs/lr, mirroring
#                            run_experiments.sh), with checkpoint_D1.pth
#                            copied in first — so re-running this script
#                            with different hyperparameters never overwrites
#                            an earlier experiment's checkpoints/logs.
#   4. diagnose_routing.py — per-class routing diagnostics for D4
#
# Every experiment's checkpoints + logs live under their own directory:
#   checkpoints/<EXP_NAME>/checkpoint_D{1,2,3,4}.pth
#   checkpoints/<EXP_NAME>/{train,diagnose_D4}.log
# A one-line summary is appended to $RESULTS_LOG after step 3, matching the
# format used by run_experiments.sh.
# ================================================================
set -e

BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TAU_ROOT="$BASE_DIR/TAU2019_dev/TAU-urban-acoustic-scenes-2019-development"
PANNS_CKPT="$BASE_DIR/Cnn14_mAP=0.431.pth"

DATA_DIR="$BASE_DIR/tau_data_adil"
RESULTS_LOG="$BASE_DIR/results/experiment_results_tau_adil.txt"

DOMAINS="barcelona+helsinki+london+paris+stockholm+vienna;lisbon;lyon;prague"
MODEL_TYPE="bn_res_deep_clf"
ADAPTER_DEPTHS="2,2,2"
LAMBDA_CON=0.2

D1_EPOCHS=120
D1_LR=1e-3
INC_EPOCHS=180
INC_LR=1e-3
BATCH_SIZE=32
NUM_WORKERS=8

# D1 base is shared by every experiment that uses the SAME ADAPTER_DEPTHS
# (the state_dict shape depends on it) — key its folder on that so it's
# safely reusable and never silently overwritten by a differently-shaped run.
DEPTHS_TAG="$(echo "$ADAPTER_DEPTHS" | tr ',' '_')"
D1_CKPT_DIR="$BASE_DIR/checkpoints/TAU_adil_D1_base_depths${DEPTHS_TAG}"

# Each incremental experiment gets its own uniquely-named checkpoint dir so
# different hyperparameter combos never collide.
EXP_NAME="TAU_ADIL_${MODEL_TYPE}_depths${DEPTHS_TAG}_lr${INC_LR}_bs${BATCH_SIZE}_ep${INC_EPOCHS}_con${LAMBDA_CON}"
CKPT_DIR="$BASE_DIR/checkpoints/$EXP_NAME"

# Set to 1 to skip a step (e.g. re-running only the incremental step)
SKIP_PREPARE=0
SKIP_D1=0
SKIP_INCREMENTAL=0
SKIP_DIAGNOSE=0

mkdir -p "$D1_CKPT_DIR" "$CKPT_DIR"

# ── 1. Build 4-domain ADIL data (shared; independent of hyperparameters) ─
# if [ "$SKIP_PREPARE" != "1" ]; then
#     echo "========================================================"
#     echo "  [1/4] Building 4-domain ADIL manifest -> $DATA_DIR"
#     echo "========================================================"
#     python "$BASE_DIR/workshop_paper/tau2019_experiment/prepare_tau2019.py" \
#         --tau_root "$TAU_ROOT" \
#         --domains "$DOMAINS" \
#         --out "$DATA_DIR" \
#         2>&1 | tee "$CKPT_DIR/prepare.log"
# fi

# ── 2. D1 base = full unfrozen on 6 cities (nb_tasks auto=4) ────────────
#      Saved in a depths-keyed shared dir, then copied into this
#      experiment's own dir so step 3 always has its own immutable copy.
if [ "$SKIP_D1" != "1" ]; then
    echo "========================================================"
    echo "  [2/4] Training D1 base (6 pooled cities) -> $D1_CKPT_DIR"
    echo "========================================================"
    TAU_DATA_DIR="$DATA_DIR" python \
        "$BASE_DIR/workshop_paper/tau2019_experiment/train_d1_base.py" \
        --checkpoint_dir "$D1_CKPT_DIR" \
        --panns_ckpt "$PANNS_CKPT" \
        --epochs "$D1_EPOCHS" --lr "$D1_LR" \
        --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --adapter_depths "$ADAPTER_DEPTHS" \
        --cuda
    # train_d1_base.py already tees its own output to $D1_CKPT_DIR/train_d1.log
fi

cp "$D1_CKPT_DIR/checkpoint_D1.pth" "$CKPT_DIR/checkpoint_D1.pth"

# ── 3. Incremental D2 -> D3 -> D4, SYMMETRIC adapters (depth 2 each) ────
if [ "$SKIP_INCREMENTAL" != "1" ]; then
    echo "========================================================"
    echo "  [3/4] Incremental training D2 -> D3 -> D4 -> $CKPT_DIR"
    echo "========================================================"
    INC_LOG="$CKPT_DIR/train.log"
    DCASE_DATASET=tau TAU_DATA_DIR="$DATA_DIR" python \
        "$BASE_DIR/code/baseline/baseline_DIL_task7.py" train \
        --model_type "$MODEL_TYPE" --adapter_depths "$ADAPTER_DEPTHS" \
        --lambda_con "$LAMBDA_CON" --epoch "$INC_EPOCHS" \
        --checkpoint_dir "$CKPT_DIR" \
        --learning_rate "$INC_LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --cuda --augmentation none --save --class_balance \
        2>&1 | tee "$INC_LOG"
    py_exit=${PIPESTATUS[0]}

    # ── Parse D2/D3/D4/avg from the log and append to the results file ──
    d2_acc=$(grep "seen domain: \['D2'\] and its accuracy:" "$INC_LOG" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    d3_acc=$(grep "seen domain: \['D3'\] and its accuracy:" "$INC_LOG" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    d4_acc=$(grep "seen domain: \['D4'\] and its accuracy:" "$INC_LOG" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    avg_acc=$(grep "Average Accuracy:" "$INC_LOG" 2>/dev/null \
              | tail -1 | awk '{print $NF}' | tr -d ' \r\n')

    status="DONE"
    [ "$py_exit" -ne 0 ] && status="FAILED(exit=$py_exit)"

    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s | %-55s | D2=%-6s | D3=%-6s | D4=%-6s | Avg=%-6s | %s\n' \
        "$ts" "$EXP_NAME" \
        "${d2_acc:--}%" "${d3_acc:--}%" "${d4_acc:--}%" "${avg_acc:--}%" \
        "$status" >> "$RESULTS_LOG"

    echo "  [${status}] $EXP_NAME | D2=${d2_acc:--}% D3=${d3_acc:--}% D4=${d4_acc:--}% Avg=${avg_acc:--}%"
    echo "  results: $RESULTS_LOG | log: $INC_LOG"
fi

# ── 4. Diagnose any domain (nb_tasks auto=4) ─────────────────────────────
if [ "$SKIP_DIAGNOSE" != "1" ]; then
    echo "========================================================"
    echo "  [4/4] Routing diagnostics for D4"
    echo "========================================================"
    DCASE_DATASET=tau TAU_DATA_DIR="$DATA_DIR" python \
        "$BASE_DIR/workshop_paper/tau2019_experiment/diagnose_routing.py" \
        --checkpoint "$CKPT_DIR/checkpoint_D4.pth" \
        --domain D4 --adapter_depths "$ADAPTER_DEPTHS" --cuda \
        2>&1 | tee "$CKPT_DIR/diagnose_D4.log"
fi

echo ""
echo "All steps complete."
echo "  data:        $DATA_DIR"
echo "  D1 base:     $D1_CKPT_DIR  (shared across runs with depths=$ADAPTER_DEPTHS)"
echo "  experiment:  $CKPT_DIR"
echo "  logs:        $CKPT_DIR/{prepare,train,diagnose_D4}.log"
echo "  results:     $RESULTS_LOG"
