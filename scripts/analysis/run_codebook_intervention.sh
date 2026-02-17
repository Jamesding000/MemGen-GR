#!/bin/bash
# Codebook Intervention: Train different SID configs, then analyze results.

set -e

DATASET=AmazonReviews2014
CATEGORY=Beauty
SPLIT=test
MAX_HOP=4
OUTPUT_DIR=outputs
WANDB_PROJECT=codebook_density_analysis
RESULTS_DIR=logs/fine_grained_results

# CB_SIZE:N_CB:BUDGET
EXPERIMENTS=(
    "64:4:120"
    "256:4:120"
    "64:3:116"
    "256:3:116"
    "1024:3:116"
    "256:2:110"
    "1024:2:110"
    "4096:2:110"
    "1024:1:108"
    "4096:1:108"
)

sem_ids_path() {
    local CB_SIZE=$1 N_CB=$2
    local SIZES
    SIZES=$(python3 -c "print(','.join(['$CB_SIZE'] * $N_CB))")
    echo "cache/${DATASET}/${CATEGORY}/processed/sentence-t5-base_${SIZES}.sem_ids"
}

eval_results_path() {
    local CB_SIZE=$1 N_CB=$2
    echo "${RESULTS_DIR}/codebook_${CB_SIZE}x${N_CB}.csv"
}

# 1. Train
for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r CB_SIZE N_CB BUDGET <<< "$entry"
    python main.py \
        --model=TIGER \
        --dataset="$DATASET" \
        --category="$CATEGORY" \
        --wandb_project="$WANDB_PROJECT" \
        --wandb_run_name="${CB_SIZE}x${N_CB}" \
        --rq_codebook_size="$CB_SIZE" \
        --rq_n_codebooks="$N_CB" \
        --num_layers=4 \
        --num_decoder_layers=4 \
        --d_model=128 \
        --d_ff=1024 \
        --num_heads=6 \
        --d_kv=64 \
        --eval_interval=2 \
        --patience=None \
        --rq_faiss=False \
        --epochs="$BUDGET" \
        --eval_results_file="$(eval_results_path "$CB_SIZE" "$N_CB")"
done

# 2. Analyze
SPECS=()
for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r CB_SIZE N_CB BUDGET <<< "$entry"
    SPECS+=("${CB_SIZE}x${N_CB}:$(eval_results_path "$CB_SIZE" "$N_CB"):$(sem_ids_path "$CB_SIZE" "$N_CB"):${BUDGET}")
done

python analysis/codebook_intervention.py \
    --dataset="$DATASET" \
    --category="$CATEGORY" \
    --split="$SPLIT" \
    --max_hop="$MAX_HOP" \
    --output_dir="$OUTPUT_DIR" \
    --experiments "${SPECS[@]}"