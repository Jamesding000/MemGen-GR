#!/bin/bash

set -e

DATASET=AmazonReviews2014
CATEGORY=Sports_and_Outdoors
SPLIT=test
OUTPUT_DIR=outputs
N_BINS=5

DATASET_ID="${DATASET}-${CATEGORY}"

echo "============================================"
echo "  MSP Indicator Validation"
echo "  Dataset: ${DATASET_ID}"
echo "  Split:   ${SPLIT}"
echo "============================================"

python adaptive_ensemble/indicator_validation.py \
    --datasets "${DATASET_ID}" \
    --split "${SPLIT}" \
    --labels_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_bins "${N_BINS}"

echo "Done."
