#!/bin/bash

set -e

DATASET=AmazonReviews2014
CATEGORY=Sports_and_Outdoors
SPLIT=test
OUTPUT_DIR=outputs
N_MSP_BINS=4

DATASET_ID="${DATASET}-${CATEGORY}"

SEM_IDS_PATH="memgen-checkpoints/semantic_ids/${DATASET_ID}_sentence-t5-base_256,256,256,256.sem_ids"
TIGER_INFER_PATH="logs/inference_results/TIGER_${DATASET_ID}_${SPLIT}_inference_results.csv"
SASREC_INFER_PATH="logs/inference_results/SASRec_${DATASET_ID}_${SPLIT}_inference_results.csv"

echo "============================================"
echo "  Performance Analysis"
echo "  Dataset: ${DATASET_ID}"
echo "  Split:   ${SPLIT}"
echo "============================================"

python analysis/performance_analysis.py \
    --dataset "${DATASET}" \
    --category "${CATEGORY}" \
    --sem_ids_path "${SEM_IDS_PATH}" \
    --sasrec_infer_path "${SASREC_INFER_PATH}" \
    --tiger_infer_path "${TIGER_INFER_PATH}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done."
