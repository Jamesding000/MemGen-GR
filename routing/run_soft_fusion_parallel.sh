#!/bin/bash

# run in sequential
# python -m routing.run_soft_fusion_master \
#   --max_workers 1 \
#   --base_dir Confidence \
#   --output_dir routing/results \
#   --normalization min_max \
#   --n_predictions 50 \
#   --verbose

# run in parallel
python -m routing.run_soft_fusion_parallel \
  --max_workers 7 \
  --base_dir Confidence \
  --output_dir routing/results \
  --normalization min_max \
  --n_predictions 50