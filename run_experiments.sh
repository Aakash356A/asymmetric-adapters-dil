#!/bin/bash
# ================================================================
# DCASE Task 7 – Experiment Runner
#
# Serial usage:
#   bash run_experiments.sh
#   bash run_experiments.sh 2>&1 | tee runner.log
#
# Parallel execution:
#   Experiments listed with `q` are batched MAX_PARALLEL at a time.
#   Each job writes to its own train.log; the console shows only
#   progress banners so output stays readable.
#   Use `run_experiment` (no `q`) for a single serial run.
#
# Results are appended to experiment_results.txt after every run.
# ================================================================

BASE_DIR="/home/aakashd/dcase_task7"
SCRIPT="$BASE_DIR/code/baseline/baseline_DIL_task7.py"
D1_CKPT="$BASE_DIR/pretrained_models/checkpoint_D1.pth"
RESULTS_LOG="$BASE_DIR/experiment_results.txt"

# ── Parallel settings ────────────────────────────────────────────────────
MAX_PARALLEL=3   # jobs to run at once  (12 GB GPU ÷ 3 ≈ 4 GB each)
                 # → set to 2 if you see CUDA OOM errors

SER_WORKERS=16   # dataloader workers for serial runs
PAR_WORKERS=4    # dataloader workers per job in parallel mode
                 # total CPU threads = PAR_WORKERS × MAX_PARALLEL
# ────────────────────────────────────────────────────────────────────────

# ================================================================
# Core function: run one experiment
# ================================================================
# run_experiment  <model_type>  <lr>  <bs>  <epochs>  [tag]
#                 [lambda_reg=0]  [lambda_con=0]
#
# When called via `q` the env-var _PARALLEL=1 is set, which:
#   • redirects all output to train.log (no console spam)
#   • uses PAR_WORKERS instead of SER_WORKERS
# ================================================================
run_experiment() {
    local model_type=$1
    local lr=$2
    local bs=$3
    local ep=$4
    local tag="${5:-}"
    local lambda_reg="${6:-0.0}"
    local lambda_con="${7:-0.0}"
    local workers="${_PAR_WORKERS:-$SER_WORKERS}"

    # Human-readable experiment name
    local name="${model_type^^}_lr${lr}_bs${bs}_ep${ep}"
    [ -n "$tag" ] && name="${name}_${tag}"

    local ckpt_dir="$BASE_DIR/checkpoints/$name"
    local log_file="$ckpt_dir/train.log"

    # Banner (serial mode only — parallel banner printed by q())
    if [ "${_PARALLEL:-0}" != "1" ]; then
        echo ""
        echo "========================================================"
        echo "  Experiment : $name"
        echo "  lambda_reg : $lambda_reg  lambda_con : $lambda_con"
        echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================"
    fi

    mkdir -p "$ckpt_dir"
    cp "$D1_CKPT" "$ckpt_dir/checkpoint_D1.pth"

    # ── Train ────────────────────────────────────────────────────
    if [ "${_PARALLEL:-0}" = "1" ]; then
        # Parallel: all output to log file only
        python3 "$SCRIPT" train \
            --augmentation=none \
            --learning_rate="$lr" \
            --batch_size="$bs" \
            --cuda \
            --num_workers="$workers" \
            --epoch="$ep" \
            --model_type="$model_type" \
            --checkpoint_dir="$ckpt_dir" \
            --lambda_reg="$lambda_reg" \
            --lambda_con="$lambda_con" \
            --save > "$log_file" 2>&1
        local py_exit=$?
    else
        # Serial: stream to console AND log
        python3 "$SCRIPT" train \
            --augmentation=none \
            --learning_rate="$lr" \
            --batch_size="$bs" \
            --cuda \
            --num_workers="$workers" \
            --epoch="$ep" \
            --model_type="$model_type" \
            --checkpoint_dir="$ckpt_dir" \
            --lambda_reg="$lambda_reg" \
            --lambda_con="$lambda_con" \
            --save 2>&1 | tee "$log_file"
        local py_exit=${PIPESTATUS[0]}
    fi

    # ── Parse results from log ───────────────────────────────────
    local d2_acc d3_acc avg_acc
    d2_acc=$(grep "seen domain: \['D2'\] and its accuracy:" "$log_file" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    d3_acc=$(grep "seen domain: \['D3'\] and its accuracy:" "$log_file" 2>/dev/null \
             | tail -1 | awk -F': ' '{print $NF}' | tr -d ' \r\n')
    avg_acc=$(grep "Average Accuracy:" "$log_file" 2>/dev/null \
              | tail -1 | awk '{print $NF}' | tr -d ' \r\n')

    local status="DONE"
    [ "$py_exit" -ne 0 ] && status="FAILED(exit=$py_exit)"

    # ── Append one line to the shared results log ────────────────
    local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s | %-45s | D2=%-6s | D3=%-6s | Avg=%-6s | %s\n' \
        "$ts" "$name" \
        "${d2_acc:--}%" "${d3_acc:--}%" "${avg_acc:--}%" \
        "$status" >> "$RESULTS_LOG"

    echo "  [${status}] $name | D2=${d2_acc:--}% D3=${d3_acc:--}% Avg=${avg_acc:--}% | log: $log_file"
}


# ================================================================
# Parallel queue helpers
# ================================================================

_job_pids=()

# Wait for all queued jobs to finish and clear the list
_flush_jobs() {
    [ ${#_job_pids[@]} -eq 0 ] && return
    echo ""
    echo "  >>> waiting for ${#_job_pids[@]} job(s)..."
    for pid in "${_job_pids[@]}"; do
        wait "$pid"
    done
    _job_pids=()
    echo "  >>> batch done  $(date '+%H:%M:%S')"
    echo ""
}

# q — queue one experiment; auto-flushes when MAX_PARALLEL is reached.
# Syntax identical to run_experiment.
q() {
    local name="${1^^}_lr${2}_bs${3}_ep${4}${5:+_$5}"
    echo "  [+] queued: $name"
    _PARALLEL=1 _PAR_WORKERS=$PAR_WORKERS run_experiment "$@" &
    _job_pids+=($!)
    if (( ${#_job_pids[@]} >= MAX_PARALLEL )); then
        _flush_jobs
    fi
}


# ================================================================
# EXPERIMENTS
# ================================================================
# Use `q` to run in parallel batches of MAX_PARALLEL.
# Use `run_experiment` to run a single job serially (full console output).
#
# Signature:
#   q / run_experiment  <model>  <lr>  <bs>  <ep>  [tag]  [λ_reg]  [λ_con]
# ================================================================

echo "================================================================"
echo "  DCASE Task 7 – starting experiments  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  MAX_PARALLEL=$MAX_PARALLEL  PAR_WORKERS=$PAR_WORKERS"
echo "================================================================"

# ── Batch 1: ADIL head + lr sweep (3 jobs) ──────────────────────────────
echo ""
echo "--- Batch 1: ADIL / CLF heads + lr sweep ---"
q bn_res_adil  1e-3  32  120  res_adil_head       # ADIL residual classifier
q bn_res_clf   1e-3  32  120  res_clf_head         # per-domain head warm-started
q bn_res       5e-4  32  120  lr_5e4               # lr sweep down

# ── Batch 2: lr sweep + SupCon (3 jobs) ─────────────────────────────────
echo ""
echo "--- Batch 2: lr sweep + SupCon ---"
q bn_res       2e-3  32  120  lr_2e3               # lr sweep up
q bn_res       1e-3  32  120  supcon_0.1  0.0  0.1 # contrastive λ=0.1
q bn_res       1e-3  32  120  supcon_0.3  0.0  0.3 # contrastive λ=0.3

# ── Batch 3: combo + remainder (flush any leftover) ─────────────────────
echo ""
echo "--- Batch 3: ADIL + SupCon combo ---"
q bn_res_adil  1e-3  32  120  adil_supcon_0.1  0.0  0.1

_flush_jobs   # drain any remaining jobs

# ────────────────────────────────────────────────────────────────────────
# Previous / archived experiments (commented out)
# ────────────────────────────────────────────────────────────────────────
# run_experiment bn       1e-4  32  120  baseline_repro3
# run_experiment bn_clf   1e-4  32  120  perdomain_head
# run_experiment bn_cos   1e-4  32  120  perdomain_cosine
# run_experiment bn_film  1e-4  32  120  film_adapters
# run_experiment bn_se    1e-3  32  120  se_adapters
# run_experiment bn_se_deep 1e-3 32 120  deeper_se_adapters
# run_experiment bn_se_clf  1e-3 32 120  se_plus_domain_head
# run_experiment bn_res   1e-3  32  120  residual_bottleneck
# run_experiment bn_gn    1e-3  32  120  groupnorm_baseline
# run_experiment bn_gn_se 1e-3  32  120  groupnorm_plus_se
# run_experiment bn_se    1e-3  32  120  bar_lambda0.5  0.5
# run_experiment bn_se    1e-3  32  120  bar_lambda1.0  1.0

echo ""
echo "================================================================"
echo "  All experiments finished  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: $RESULTS_LOG"
echo "================================================================"
