#!/bin/bash

set -e

BASE_DIR="memgen-checkpoints"
SEM_IDS_DIR="${BASE_DIR}/semantic_ids"
OUTPUT_DIR="outputs"
N_PREDICTIONS=50
TOP_K=10

DATASETS=(
    "AmazonReviews2014:Sports_and_Outdoors:"
    "AmazonReviews2014:Beauty:"
    "AmazonReviews2023:Industrial_and_Scientific:"
    "AmazonReviews2023:Musical_Instruments:"
    "AmazonReviews2023:Office_Products:"
    "Steam::"
    "Yelp::Yelp_2020"
)

SPLITS=("val" "test")
TOTAL=${#DATASETS[@]}
CURRENT=0

mkdir -p "$OUTPUT_DIR"

for DATASET_CONFIG in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    IFS=':' read -r DATASET CATEGORY VERSION <<< "$DATASET_CONFIG"

    GRID_ARGS=(--dataset_name "$DATASET" --category "$CATEGORY" --version "$VERSION")
    INFER_ARGS=(--dataset "$DATASET" --category "$CATEGORY" --version "$VERSION")

    CKPT_SUFFIX=""
    ID_SUFFIX=""
    if [ -n "$VERSION" ]; then
        CKPT_SUFFIX="-version_${VERSION}"
        ID_SUFFIX="-${VERSION}"
    elif [ -n "$CATEGORY" ]; then
        CKPT_SUFFIX="-category_${CATEGORY}"
        ID_SUFFIX="-${CATEGORY}"
    fi

    DISPLAY_NAME="${DATASET}${ID_SUFFIX}"
    TIGER_CKPT="${BASE_DIR}/TIGER/TIGER-${DATASET}${CKPT_SUFFIX}.pth"
    SASREC_CKPT="${BASE_DIR}/SASRec/SASRec-${DATASET}${CKPT_SUFFIX}.pth"
    SEM_IDS_PATH="${SEM_IDS_DIR}/${DATASET}${ID_SUFFIX}_sentence-t5-base_256,256,256,256.sem_ids"

    echo -e "\n========================================================================="
    echo " [$CURRENT/$TOTAL] Processing: $DISPLAY_NAME"
    echo "========================================================================="

    # first run inference for both splits
    for SPLIT in "${SPLITS[@]}"; do
        echo "  -> Running Inference (${SPLIT})"

        python -m adaptive_ensemble.tiger_inference \
            "${INFER_ARGS[@]}" \
            --model_ckpt "$TIGER_CKPT" \
            --sem_ids_path "$SEM_IDS_PATH" \
            --split "$SPLIT" \
            --d_model 128 --d_ff 1024 --num_layers 4 \
            --num_decoder_layers 4 --num_heads 6 --d_kv 64 \
            --n_predictions "$N_PREDICTIONS"

        python -m adaptive_ensemble.sasrec_inference \
            "${INFER_ARGS[@]}" \
            --checkpoint_path "$SASREC_CKPT" \
            --eval "$SPLIT" \
            --n_predictions "$N_PREDICTIONS"
    done

    # run the grid search for the adaptive ensemble and baselines
    echo "  -> Running Adaptive Ensemble Evaluation"
    python -m adaptive_ensemble.grid_search \
        "${GRID_ARGS[@]}" \
        --base_dir "$OUTPUT_DIR" \
        --top_k "$TOP_K" \
        --n_predictions "$N_PREDICTIONS"

done

echo -e "\n========================================================================="
echo " ✅ All datasets processed successfully. Results saved to: $OUTPUT_DIR"
echo "========================================================================="