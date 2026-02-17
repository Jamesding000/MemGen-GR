#!/bin/bash

set -e

BASE_DIR="saved_models"
SEM_IDS_DIR="${BASE_DIR}/semantic_ids"
N_PREDICTIONS=50

# Format: "dataset_name:category:version"
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

# 1. Generate inference results for SASRec and TIGER
for DATASET_CONFIG in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    IFS=':' read -r DATASET CATEGORY VERSION <<< "$DATASET_CONFIG"

    # Construct paths and args based on dataset type
    if [ -n "$VERSION" ]; then
        CKPT_SUFFIX="-version_${VERSION}"
        ID_SUFFIX="-${VERSION}"
        ARGS="--dataset $DATASET --version $VERSION"
    elif [ -n "$CATEGORY" ]; then
        CKPT_SUFFIX="-category_${CATEGORY}"
        ID_SUFFIX="-${CATEGORY}"
        ARGS="--dataset $DATASET --category $CATEGORY"
    else
        CKPT_SUFFIX=""
        ID_SUFFIX=""
        ARGS="--dataset $DATASET"
    fi

    TIGER_CKPT="${BASE_DIR}/TIGER-${DATASET}${CKPT_SUFFIX}.pth"
    SASREC_CKPT="${BASE_DIR}/SASRec-${DATASET}${CKPT_SUFFIX}.pth"
    
    # Handle filename formatting differences for Semantic IDs if necessary
    # Assuming pattern: Dataset-Suffix_sentence-t5...
    SEM_IDS_PATH="${SEM_IDS_DIR}/${DATASET}${ID_SUFFIX}_sentence-t5-base_256,256,256,256.sem_ids"
    
    # Check if files exist to avoid silent failures
    if [ ! -f "$TIGER_CKPT" ]; then echo "Warning: TIGER ckpt not found: $TIGER_CKPT"; fi
    if [ ! -f "$SASREC_CKPT" ]; then echo "Warning: SASRec ckpt not found: $SASREC_CKPT"; fi

    echo "Processing [$CURRENT/$TOTAL]: $DATASET ($CKPT_SUFFIX)"

    for SPLIT in "${SPLITS[@]}"; do
        echo "  -> Running Split: $SPLIT"

        # 1. TIGER Inference
        python adaptive_ensemble/tiger_inference.py \
            $ARGS \
            --model_ckpt "$TIGER_CKPT" \
            --sem_ids_path "$SEM_IDS_PATH" \
            --split "$SPLIT" \
            --d_model 128 \
            --d_ff 1024 \
            --num_layers 4 \
            --num_decoder_layers 4 \
            --num_heads 6 \
            --d_kv 64 \
            --n_predictions "$N_PREDICTIONS"

        # 2. SASRec Confidence Extraction
        python adaptive_ensemble/sasrec_inference.py \
            $ARGS \
            --checkpoint "$SASREC_CKPT" \
            --eval "$SPLIT" \
            --n_predictions "$N_PREDICTIONS"
    done
done

# 2. Run adaptive ensemble grid search in parallel
DATASET_IDS=()
for DATASET_CONFIG in "${DATASETS[@]}"; do
    IFS=':' read -r DS CAT VER <<< "$DATASET_CONFIG"
    if [ -n "$VER" ]; then
        DATASET_IDS+=("${DS}-${VER}")
    elif [ -n "$CAT" ]; then
        DATASET_IDS+=("${DS}-${CAT}")
    else
        DATASET_IDS+=("${DS}")
    fi
done

echo "============================================================"
echo "Running adaptive ensemble grid search..."
echo "============================================================"

python -m adaptive_ensemble.grid_search \
    --datasets "${DATASET_IDS[@]}" \
    --max_workers 7 \
    --base_dir outputs \
    --output_dir outputs \
    --normalization min_max \
    --n_predictions "$N_PREDICTIONS"
