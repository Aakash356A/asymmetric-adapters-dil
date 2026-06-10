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

BASE_DIR="/home/aakash/dcase_task7"
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
    local con_temp="${8:-0.07}"
    local adapter_depths="${9:-}"        # e.g. "1,3" for bn_res_deep (D2,D3)
    local augment="${10:-0}"             # 1 = enable waveform augmentation
    local class_balance="${11:-0}"       # 1 = class-balanced sampler
    local workers="${_PAR_WORKERS:-$SER_WORKERS}"

    # Optional flags appended only when set
    local extra_flags=""
    [ -n "$adapter_depths" ]   && extra_flags="$extra_flags --adapter_depths=$adapter_depths"
    [ "$augment" = "1" ]       && extra_flags="$extra_flags --augment"
    [ "$class_balance" = "1" ] && extra_flags="$extra_flags --class_balance"

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
        echo "  lambda_reg : $lambda_reg  lambda_con : $lambda_con  con_temp : $con_temp"
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
            --con_temp="$con_temp" \
            $extra_flags \
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
            --con_temp="$con_temp" \
            $extra_flags \
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
    local lcon="${7:-0.0}"
    local temp="${8:-0.07}"
    local depths="${9:-}"
    local aug="${10:-0}"
    local bal="${11:-0}"
    local note=""
    [ -n "$depths" ] && note="$note  depths=$depths"
    [ "$aug" = "1" ] && note="$note  +augment"
    [ "$bal" = "1" ] && note="$note  +balance"
    echo "  [+] queued: $name  λ_con=$lcon  τ=$temp$note"
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

#          model_type  lr      bs    epochs   [tag]

# --- baseline (always keep one as the reference point) -------------------
# run_experiment bn       1e-4    32    120      baseline_repro3

# --- NEW architecture experiments (idea #1-#4) --------------------------
# All use the best-known baseline recipe (lr=1e-4, bs=32, ep=120) so the
# only thing that changes vs. the baseline is the architecture.
# run_experiment bn_clf   1e-4    32    120      perdomain_head        # #1 per-domain linear head
# run_experiment bn_cos   1e-4    32    120      perdomain_cosine      # #2 per-domain cosine/NCM head
# run_experiment bn_film  1e-4    32    120      film_adapters         # #3a per-domain FiLM adapters
# run_experiment bn_se_freeze    1e-4    32    120      se_freeze_backbone           # #3b per-domain SE adapters
# run_experiment bn_att   1e-4    32    120      attention_pooling     # #4  per-domain attention pooling

# --- new experiments to close D2→D3 gap ---------------------------------
# run_experiment bn_se_deep  1e-3    32    120      deeper_se_adapters    # A: 2-layer SE bottleneck
# run_experiment bn_se_clf   1e-3    32    120      se_plus_domain_head   # B: SE adapters + per-domain head
# run_experiment bn_res      1e-3    32    120      residual_bottleneck   # C: zero-init residual conv adapters

# --- GroupNorm variants --------------------------------------------------
# run_experiment bn_gn     1e-3  32  120  groupnorm_baseline   # GN only, no adapters
# run_experiment bn_gn_se  1e-3  32  120  groupnorm_plus_se    # GN + SE adapters

# --- Boundary-Aware Regularization (PACE) on best model so far ----------
#          model    lr     bs   ep   tag                    lambda_reg
# run_experiment bn_se  1e-3  32  120  bar_lambda0.5           0.5
# run_experiment bn_se  1e-3  32  120  bar_lambda1.0           1.0

# # --- GroupNorm variants --------------------------------------------------
# run_experiment bn_gn     1e-3  32  120  groupnorm_baseline   # GN only, no adapters
# run_experiment bn_gn_se  1e-3  32  120  groupnorm_plus_se    # GN + SE adapters

# =====================================================================
# NEW EXPERIMENTS — building on BN_RES (current best: D2=73.87% D3=53.47%)
# =====================================================================

# --- 1. ADIL residual head on BN_RES ---------------------------------
# logits = G1(emb) + Gt(emb)  — shared head + zero-init domain correction
#          model        lr     bs   ep   tag
# run_experiment bn_res_adil  1e-3  32  120  res_adil_head

# # --- 2. Per-domain head on BN_RES (warm-started, replaces G1) --------
# run_experiment bn_res_clf   1e-3  32  120  res_clf_head

# # --- 3. BN_RES learning rate sweep -----------------------------------
# # Best lr for residual adapters might not be 1e-3
# q bn_res  5e-4  32  120  lr_5e4
# q bn_res  2e-3  32  120  lr_2e3

# --- 4. BN_RES + Supervised Contrastive Loss -------------------------
# Pulls same-class embeddings together, pushes different-class apart
# lambda_reg=0  lambda_con=X
#          model   lr     bs   ep   tag              lreg  lcon
# q bn_res  1e-3  32  120  supcon_0.1     0.0   0.1
# q bn_res  1e-3  32  120  supcon_0.3     0.0   0.3

# --- 5. BN_RES_ADIL + SupCon (best of both) --------------------------
# q bn_res_adil  1e-3  32  120  adil_supcon_0.1  0.0  0.1
# _flush_jobs

# =====================================================================
# BATCH 4 — Combine two leaders: BN_RES_CLF + SupCon
#
# Leaders so far:
#   BN_RES_CLF          D2=77.31%  D3=53.97%  Avg=65.64%
#   BN_RES + SupCon 0.1 D2=77.62%  D3=53.72%  Avg=65.67%
#   BN_RES + SupCon 0.3 D2=74.49%  D3=55.71%  Avg=65.10%  (best D3)
#
# Hypothesis: CLF gives a better per-domain classifier boundary,
# SupCon tightens the embedding clusters — combining them should
# push both D2 and D3 simultaneously.
# =====================================================================

# echo ""
# echo "--- Batch 4: BN_RES_CLF + SupCon combos ---"
# #          model        lr    bs  ep   tag                  lreg  lcon
# q bn_res_clf  1e-3  32  120  clf_supcon_0.1          0.0   0.1
# q bn_res_clf  1e-3  32  120  clf_supcon_0.2          0.0   0.2

# echo ""
# echo "--- Batch 5: SupCon sweet-spot on base BN_RES ---"
# q bn_res_clf  1e-3  32  120  clf_supcon_0.3          0.0   0.3
# q bn_res      1e-3  32  120  supcon_0.2              0.0   0.2

# echo ""
# echo "--- Batch 6: DTN2 (LayerNorm-style norm + tanh gate) ---"
# dtn2 replaces the raw-activation DTN with a normalise-first variant:
#   y = gamma * tanh(alpha * norm_C(x)) + beta
# Should be more stable than plain DTN when conv output magnitudes vary.
#          model   lr     bs  ep   tag
# q dtn2  1e-3  32  120  layernorm_tanh_gate

# _flush_jobs

# =====================================================================
# BATCH 6 — Fine-tune SupCon on BN_RES (current best: 66.70%)
#
# Three axes to explore:
#   A. Lambda fine-tuning  (λ=0.15 and λ=0.25)
#   B. Temperature tuning  (τ=0.05 and τ=0.10, with best λ=0.2)
#   C. Larger batch size   (bs=64 for more contrastive pairs per batch)
# =====================================================================

# echo ""
# echo "--- Batch 6: SupCon fine-tuning on BN_RES ---"
# #        model   lr    bs   ep   tag               lreg  lcon  temp
# q bn_res  1e-3  32  120  supcon_0.15         0.0   0.15  0.07
# q bn_res  1e-3  32  120  supcon_0.25         0.0   0.25  0.07
# q bn_res  1e-3  32  120  supcon_temp0.05     0.0   0.2   0.05

# echo ""
# echo "--- Batch 7: temperature + big batch ---"
# q bn_res  1e-3  32  120  supcon_temp0.10     0.0   0.2   0.10
# q bn_res  1e-3  64  120  supcon_bs64         0.0   0.2   0.07

# _flush_jobs

# run_experiment bn_grn  1e-3  32  120  grn_norm

# =====================================================================
# BATCH 8 — Per-domain adapter DEPTH  (bn_res_deep)  + SupCon λ=0.2
#
# Tests "does D3 want more conv capacity than D2?".  Adapters are
# per-domain & zero-init, so D2 is never affected and identity-init is safe.
# Recipe = current best (lr=1e-3, bs=32, SupCon λ=0.2, τ=0.07).
#
#   depths "D2,D3":
#     sym_deep2      2,2  -> both domains 2 adapters/block (capacity baseline)
#     asym_d3        1,2  -> D3 gets +1 adapter/block      (THE key run)
#     asym_d3_heavy  1,3  -> D3 gets +2 adapters/block
#
# q args:  model  lr  bs  ep  tag  lreg lcon temp  depths  aug bal
# =====================================================================
# echo ""
# echo "--- Batch 8: per-domain adapter depth (bn_res_deep) ---"
# q bn_res_deep  1e-3  32  120  sym_deep2      0.0  0.2  0.07  "2,2"  0  0
# q bn_res_deep  1e-3  32  120  asym_d3        0.0  0.2  0.07  "1,2"  0  0
# q bn_res_deep  1e-3  32  120  asym_d3_heavy  0.0  0.2  0.07  "1,3"  0  0
# _flush_jobs

# =====================================================================
# BATCH 9 — Data augmentation + class balance on the best recipe
#
# Uses AudioAugmenter (waveform-level) and/or a class-balanced sampler
# on bn_res + SupCon λ=0.2.  Isolates the effect of DATA, not architecture.
#
# q args:  model  lr  bs  ep  tag  lreg lcon temp  depths  aug bal
# =====================================================================
# echo ""
# echo "--- Batch 9: augmentation / class balance on BN_RES ---"
# q bn_res  1e-3  32  120  aug            0.0  0.2  0.07  ""  1  0
# q bn_res  1e-3  32  120  aug_balanced   0.0  0.2  0.07  ""  1  1
# q bn_res  1e-3  32  120  balanced       0.0  0.2  0.07  ""  0  1
# _flush_jobs

# =====================================================================
# ANGAD ROUTER — D1-BN embedding + sigmoid rejection router
#
# Separate from run_experiment (own script).  Step 1 trains the router
# (only needs the D1 checkpoint).  Step 2 evaluates routed classification
# using the FINAL checkpoint of the best trained model.
#
# Pick BEST_RUN = the bn_res + SupCon 0.2 run dir (its checkpoint_D3.pth
# holds the trained D2/D3 params).
# ---------------------------------------------------------------------
# ROUTER_DIR="$BASE_DIR/checkpoints/router"
# BEST_RUN="$BASE_DIR/checkpoints/BN_RES_lr1e-3_bs32_ep120_supcon_0.2"
# mkdir -p "$ROUTER_DIR"
# #
# python3 "$BASE_DIR/code/baseline/domain_router.py" train \
#     --model_type bn_res --learning_rate 1e-3 --batch_size 32 \
#     --epoch 30 --num_workers "$SER_WORKERS" --cuda \
#     --d1_checkpoint "$D1_CKPT" \
#     --router_out "$ROUTER_DIR/router.pth"
# #
# python3 "$BASE_DIR/code/baseline/domain_router.py" eval \
#     --model_type bn_res --num_workers "$SER_WORKERS" --cuda \
#     --model_checkpoint "$BEST_RUN/checkpoint_D3.pth" \
#     --router_in "$ROUTER_DIR/router.pth" --threshold 0.5

# # --- extra sweeps to try once you see which idea wins -------------------
# run_experiment bn_clf   5e-4    32    120      perdomain_head_lr5e4
# run_experiment bn_cos   5e-4    32    120      perdomain_cosine_lr5e4
# run_experiment bn_film  1e-4    32    150      film_adapters_ep150

# =====================================================================
# BATCH 10 — bn_res_deep_clf: deep adapters + per-domain head + supcon
#
# Best D2 recipe: bn_res_clf + supcon_0.2  (D2=77.93%)
# Best D3 recipe: bn_res_deep asym_d3_heavy depth 1,3  (D3=58.93%)
# This batch stacks both: deep adapters AND a clf head.
#
# Hypothesis: clf head fixes D2 ceiling; deep D3 adapters fix D3 ceiling.
# Run them separately (D2-only depth vs D3-only depth) and together.
#
# q args:  model              lr    bs  ep   tag                   lreg  lcon  temp  depths  aug bal
# =====================================================================
# echo ""
# echo "--- Batch 10: bn_res_deep_clf (deep adapters + clf head + supcon) ---"
# q bn_res_deep_clf  1e-3  32  120  deep_clf_d3_sup02        0.0   0.2  0.07  "1,3"  0  0
# q bn_res_deep_clf  1e-3  32  120  deep_clf_sym2_sup02      0.0   0.2  0.07  "2,2"  0  0
# _flush_jobs

# q bn_res_deep_clf  1e-3  64  120  deep_clf_d3_bs64_sup02   0.0   0.2  0.07  "1,3"  0  0
# q bn_res_deep_clf  1e-3  32  120  deep_clf_d3_sup01        0.0   0.1  0.07  "1,3"  0  0
# _flush_jobs

# =====================================================================
# BATCH 11 — Extended training (180 epochs) on best recipe
#
# The cosine LR schedule already decays to 1e-7; giving it more epochs
# lets it descend further without changing any other hyperparameter.
# Run on both base BN_RES_CLF supcon and the new deep_clf.
#
# q args:  model              lr    bs  ep   tag                   lreg  lcon  temp  depths  aug bal
# =====================================================================
# echo ""
# echo "--- Batch 11: longer training (180 ep) ---"
# q bn_res_clf       1e-3  32  180  clf_sup02_ep180          0.0   0.2  0.07  ""     0  0
# q bn_res_deep_clf  1e-3  32  180  deep_clf_d3_ep180        0.0   0.2  0.07  "1,3"  0  0
# _flush_jobs

# =====================================================================
# BATCH 12 — TTA evaluation on best existing checkpoints

# No training needed — just re-run inference with tta_n=8.
# Uses --resume to skip training and go straight to acc_prev.
# Fill in BEST_RES_CLF and BEST_RES_DEEP with your actual checkpoint dirs.
# =====================================================================
BEST_RES_CLF="$BASE_DIR/checkpoints/BN_RES_CLF_lr1e-3_bs32_ep120_clf_supcon_0.2"
BEST_RES_DEEP="$BASE_DIR/checkpoints/BN_RES_DEEP_lr1e-3_bs32_ep120_asym_d3_heavy"

echo ""
echo "--- Batch 12: TTA (tta_n=8) on top checkpoints ---"
python3 "$SCRIPT" train \
    --augmentation=none --learning_rate=1e-3 --batch_size=32 --epoch=1 \
    --model_type=bn_res_clf --checkpoint_dir="$BEST_RES_CLF" \
    --lambda_con=0.2 --tta_n=8 --resume --cuda --num_workers=$SER_WORKERS --save

python3 "$SCRIPT" train \
    --augmentation=none --learning_rate=1e-3 --batch_size=32 --epoch=1 \
    --model_type=bn_res_deep --checkpoint_dir="$BEST_RES_DEEP" \
    --adapter_depths=1,3 --lambda_con=0.2 --tta_n=8 --resume --cuda --num_workers=$SER_WORKERS --save

# =====================================================================
# BATCH 13 — Ensemble of best D2 and D3 checkpoints
#
# Uncomment once BEST_RES_CLF and BEST_RES_DEEP are available.
# Averages the two models' softmax predictions per domain.
# =====================================================================
echo ""
echo "--- Batch 13: ensemble eval ---"
python3 "$SCRIPT" ensemble_eval \
    --ckpt_dirs="$BEST_RES_CLF,$BEST_RES_DEEP" \
    --model_types="bn_res_clf,bn_res_deep" \
    --adapter_depths="1,3" \
    --tta_n=4 --cuda --num_workers=$SER_WORKERS


