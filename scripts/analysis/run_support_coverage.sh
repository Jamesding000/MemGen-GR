#!/bin/bash

set -e

DATASET=AmazonReviews2014
CATEGORY=Sports_and_Outdoors
SPLIT=test
MAX_HOP=4
OUTPUT_DIR=outputs

DATASET_ID="${DATASET}-${CATEGORY}"

SEM_IDS_PATH="saved_models/semantic_ids/${DATASET_ID}_sentence-t5-base_256,256,256,256.sem_ids"
TIGER_INFER_PATH="logs/inference_results/TIGER_${DATASET_ID}_${SPLIT}_inference_results.csv"
SASREC_INFER_PATH="logs/inference_results/SASRec_${DATASET_ID}_${SPLIT}_inference_results.csv"

echo "============================================"
echo "  Support Coverage Analysis"
echo "  Dataset: ${DATASET_ID}"
echo "  Split:   ${SPLIT}"
echo "============================================"

python analysis/support_coverage.py \
    --dataset "${DATASET}" \
    --category "${CATEGORY}" \
    --sem_ids_path "${SEM_IDS_PATH}" \
    --tiger_infer_path "${TIGER_INFER_PATH}" \
    --sasrec_infer_path "${SASREC_INFER_PATH}" \
    --split "${SPLIT}" \
    --max_hop "${MAX_HOP}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done."
