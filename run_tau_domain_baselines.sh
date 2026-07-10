#!/bin/bash
# ================================================================
# TAU2019 standalone per-domain sanity check
#
# Trains FOUR independent, fully fine-tuned classifiers, one per domain,
# with NO domain-incremental sequencing, NO frozen shared backbone, and NO
# per-domain adapters/heads -- each city (or pooled city group) gets its own
# model trained from PANNs, so you can see how learnable each domain's data
# is on its own before running the DIL pipeline (train_d1_base.py +
# baseline_DIL_task7.py train).
#
# Domains (from prepare_tau2019.py --domains
#   "barcelona+helsinki+london+paris+stockholm+vienna;lisbon;lyon;prague"):
#   D1 = 6 pooled cities, D2 = lisbon, D3 = lyon, D4 = prague
#
# Usage:
#   bash run_tau_domain_baselines.sh
#   bash run_tau_domain_baselines.sh 2>&1 | tee runner_standalone.log
#
# Each domain gets its own checkpoint dir + log file:
#   checkpoints/TAU_standalone_<domain>_<label>/checkpoint_<domain>.pth
#   checkpoints/TAU_standalone_<domain>_<label>/train_<domain>.log
# A one-line summary per domain is appended to $RESULTS_LOG.
# ================================================================
set -e

BASE_DIR="/home/aakash/dcase_task7"
PANNS_CKPT="$BASE_DIR/Cnn14_mAP=0.431.pth"
DATA_DIR="$BASE_DIR/tau_data_adil"
RESULTS_LOG="$BASE_DIR/experiment_results_tau_standalone.txt"

EPOCHS=120
LR=1e-3
BATCH_SIZE=32
NUM_WORKERS=8

# domain:label pairs -- edit/comment out to run a subset.
DOMAINS=(
    "D1:6cities"
    "D2:lisbon"
    "D3:lyon"
    "D4:prague"
)

for entry in "${DOMAINS[@]}"; do
    domain="${entry%%:*}"
    label="${entry##*:}"
    ckpt_dir="$BASE_DIR/checkpoints/TAU_standalone_${domain}_${label}"
    log_file="$ckpt_dir/train_${domain}.log"
    mkdir -p "$ckpt_dir"

    echo "========================================================"
    echo "  Standalone domain: $domain ($label)"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    TAU_DATA_DIR="$DATA_DIR" python \
        "$BASE_DIR/workshop_paper/tau2019_experiment/train_domain_standalone.py" \
        --domain "$domain" \
        --checkpoint_dir "$ckpt_dir" \
        --panns_ckpt "$PANNS_CKPT" \
        --epochs "$EPOCHS" --lr "$LR" \
        --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --cuda
    # train_domain_standalone.py already tees its own output to $log_file

    best_acc=$(grep "^Done\. Best" "$log_file" 2>/dev/null \
               | tail -1 | grep -oP '(?<=test acc )[0-9.]+(?=%)')

    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s | %-6s | %-10s | best_acc=%-6s | ckpt_dir=%s\n' \
        "$ts" "$domain" "$label" "${best_acc:--}%" "$ckpt_dir" >> "$RESULTS_LOG"

    echo "  [DONE] $domain ($label) | best_acc=${best_acc:--}% | log: $log_file"
    echo ""
done

echo "All standalone domain runs complete. Results: $RESULTS_LOG"
cat "$RESULTS_LOG"
