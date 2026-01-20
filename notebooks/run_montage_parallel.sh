#!/bin/bash
# Parallel Montage Experiments - Each CP-Cov target runs on separate core
# Usage: ./run_montage_parallel.sh

cd "$(dirname "$0")"

OUTPUT_DIR="wfinstances_montage_results"
mkdir -p "$OUTPUT_DIR"

NUM_ITERATIONS=3000000
NUM_TRACES=300

# 8 CP-Cov targets
CP_COV_TARGETS=(0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.95)

echo "========================================================================"
echo "PARALLEL MONTAGE EXPERIMENTS"
echo "========================================================================"
echo "CP-Cov targets: ${CP_COV_TARGETS[*]}"
echo "Iterations: $NUM_ITERATIONS"
echo "Traces: $NUM_TRACES base + targeted"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate conda
source ~/.bash_profile 2>/dev/null
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate hpo_inference 2>/dev/null

# Start all experiments in parallel
PIDS=()
for cp in "${CP_COV_TARGETS[@]}"; do
    LOG_FILE="$OUTPUT_DIR/log_cp${cp//./_}.txt"
    echo "Starting CP-Cov=$cp -> $LOG_FILE"
    
    python -u wfinstances_montage_experiments.py \
        --num_traces $NUM_TRACES \
        --num_iterations $NUM_ITERATIONS \
        --cp_cov_targets $cp \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "Started ${#PIDS[@]} parallel jobs with PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/log_cp*.txt"
echo ""
echo "Check status with:"
echo "  ps aux | grep wfinstances"
echo ""

# Wait for all to complete
echo "Waiting for all experiments to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
