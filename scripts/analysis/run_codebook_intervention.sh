#!/usr/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate GenRec

DATASET="AmazonReviews2014"
CATEGORY="Beauty"
NUM_GPUS=4
SWEEP_CONFIG="analysis/sweep.yaml"
LOG_DIR="logs/sweep_agents"

mkdir -p outputs logs/fine_grained_results "$LOG_DIR"

# initialize wandb sweep and automatically get sweep id
echo "[1/3] Initializing WandB Sweep..."
SWEEP_ID=$(
  wandb sweep "$SWEEP_CONFIG" 2>&1 \
  | awk '/Run sweep agent with:/ {for (i=1; i<=NF; i++) if ($i ~ /^wandb$/ && $(i+1) == "agent") print $(i+2); exit}'
)

if [ -z "$SWEEP_ID" ]; then 
    echo "ERROR: Failed to extract SWEEP_ID."
    exit 1
fi
echo "Created Sweep: $SWEEP_ID"

# launch wandb agents in the background
echo "[2/3] Launching $NUM_GPUS agents in the background..."
for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i wandb agent "$SWEEP_ID" > "$LOG_DIR/agent_$i.log" 2>&1 &
    echo "  -> Started agent on GPU $i (PID: $!)"
done

# wait for all background wandb agents to finish
wait
echo "All sweep agents have finished."

# run the analysis
echo "[3/3] Running codebook intervention analysis..."
python analysis/codebook_intervention.py \
    --dataset="$DATASET" \
    --category="$CATEGORY" \
    --split="test" \
    --max_hop=4 \
    --output_dir="outputs" \
    --sweep_config="$SWEEP_CONFIG"

echo "Pipeline completed successfully!"