#!/bin/bash
# ================================================================
# DCASE Task 7 – Experiment Runner
# Usage:  bash run_experiments.sh
#         bash run_experiments.sh 2>&1 | tee runner.log
#
# Results are appended to experiment_results.txt after each run.
# Each experiment gets its own checkpoint dir and train.log.
# ================================================================

BASE_DIR="/home/aakashd/dcase_task7"
SCRIPT="$BASE_DIR/code/baseline/baseline_DIL_task7.py"
D1_CKPT="$BASE_DIR/pretrained_models/checkpoint_D1.pth"
RESULTS_LOG="$BASE_DIR/experiment_results.txt"
NUM_WORKERS=16

# ----------------------------------------------------------------
# run_experiment  <model_type>  <lr>  <batch_size>  <epochs>  [tag]  [lambda_reg]
#
#   model_type : bn | bn_se | bn_se_deep | bn_se_clf | bn_res | ...
#   tag        : optional free-text suffix for the folder name
#   lambda_reg : boundary-aware regularization weight (default 0 = off)
# ----------------------------------------------------------------
run_experiment() {
    local model_type=$1
    local lr=$2
    local bs=$3
    local ep=$4
    local tag="${5:-}"
    local lambda_reg="${6:-0.0}"

    # Build a human-readable experiment name
    local name="${model_type^^}_lr${lr}_bs${bs}_ep${ep}"
    [ -n "$tag" ] && name="${name}_${tag}"

    local ckpt_dir="$BASE_DIR/checkpoints/$name"
    local log_file="$ckpt_dir/train.log"

    echo ""
    echo "========================================================"
    echo "  Experiment : $name"
    echo "  lambda_reg : $lambda_reg"
    echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    mkdir -p "$ckpt_dir"

    # The pretrained D1 checkpoint must live in the experiment dir
    # so load_checkpoint() can find checkpoint_D1.pth
    cp "$D1_CKPT" "$ckpt_dir/checkpoint_D1.pth"

    # Run training; stream output to console AND capture to log
    python3 "$SCRIPT" train \
        --augmentation=none \
        --learning_rate="$lr" \
        --batch_size="$bs" \
        --cuda \
        --num_workers="$NUM_WORKERS" \
        --epoch="$ep" \
        --model_type="$model_type" \
        --checkpoint_dir="$ckpt_dir" \
        --lambda_reg="$lambda_reg" \
        --save 2>&1 | tee "$log_file"
    local py_exit="${PIPESTATUS[0]}"

    # ── Parse final per-domain accuracies from the captured log ──
    # The script prints:
    #   seen domain: ['D2'] and its accuracy: 54.7
    #   seen domain: ['D3'] and its accuracy: 35.0
    #   Average Accuracy:  44.85
    # We take the LAST occurrence of each (printed after D3 training).
    local d2_acc d3_acc avg_acc
    d2_acc=$(grep "seen domain: \['D2'\] and its accuracy:" "$log_file" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    d3_acc=$(grep "seen domain: \['D3'\] and its accuracy:" "$log_file" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    avg_acc=$(grep "Average Accuracy:" "$log_file" 2>/dev/null \
              | tail -1 | awk '{print $NF}' | tr -d ' \r\n')

    local status="DONE"
    [ "$py_exit" -ne 0 ] && status="FAILED(exit=$py_exit)"

    # Append one line to the results log
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s | %-45s | D2=%-6s | D3=%-6s | Avg=%-6s | %s\n' \
        "$ts" "$name" \
        "${d2_acc:--}%" "${d3_acc:--}%" "${avg_acc:--}%" \
        "$status" >> "$RESULTS_LOG"

    echo ""
    echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  D2=${d2_acc:-N/A}%  D3=${d3_acc:-N/A}%  Avg=${avg_acc:-N/A}%  Status=$status"
    echo "  Log      : $log_file"
    echo "  Results  : $RESULTS_LOG"
}

# ================================================================
# Define your experiments here.
# Uncomment / add lines as needed.
# ================================================================

#          model_type  lr      bs    epochs   [tag]

# --- baseline (always keep one as the reference point) -------------------
# run_experiment bn       1e-4    32    120      baseline_repro3

# --- NEW architecture experiments (idea #1-#4) --------------------------
# All use the best-known baseline recipe (lr=1e-4, bs=32, ep=120) so the
# only thing that changes vs. the baseline is the architecture.
# run_experiment bn_clf   1e-4    32    120      perdomain_head        # #1 per-domain linear head
# run_experiment bn_cos   1e-4    32    120      perdomain_cosine      # #2 per-domain cosine/NCM head
# run_experiment bn_film  1e-4    32    120      film_adapters         # #3a per-domain FiLM adapters
run_experiment bn_se_freeze    1e-4    32    120      se_freeze_backbone           # #3b per-domain SE adapters
# run_experiment bn_att   1e-4    32    120      attention_pooling     # #4  per-domain attention pooling

# --- new experiments to close D2→D3 gap ---------------------------------
run_experiment bn_se_deep  1e-3    32    120      deeper_se_adapters    # A: 2-layer SE bottleneck
run_experiment bn_se_clf   1e-3    32    120      se_plus_domain_head   # B: SE adapters + per-domain head
run_experiment bn_res      1e-3    32    120      residual_bottleneck   # C: zero-init residual conv adapters

# --- Boundary-Aware Regularization (PACE) on best model so far ----------
#          model    lr     bs   ep   tag                    lambda_reg
run_experiment bn_se  1e-3  32  120  bar_lambda0.5           0.5
run_experiment bn_se  1e-3  32  120  bar_lambda1.0           1.0

# --- extra sweeps to try once you see which idea wins -------------------
# run_experiment bn_clf   5e-4    32    120      perdomain_head_lr5e4
# run_experiment bn_cos   5e-4    32    120      perdomain_cosine_lr5e4
# run_experiment bn_film  1e-4    32    150      film_adapters_ep150


