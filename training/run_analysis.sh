#!/bin/bash

set -e 
PYTHON_EXE="python3" 
SCRIPT_PATH="./motep_jax_analysis.py" 

FOLDER_NAME="" 
RUN_NAME=""
OUTPUT_NAME="train_report"
THRESHOLD_LOSS=1e-07
MIN_STEPS=10
LR_START=3e-1
TRANSITION_STEPS=50
DECAY_RATE=1.0
GLOBAL_NORM_CLIP=9.9e-01
BATCH_SIZE=5
SCALING=1.0
MAX_DIST=5.0
MIN_DIST=0.5


CMD="$PYTHON_EXE $SCRIPT_PATH"

CMD+=" --folder_name \"$FOLDER_NAME\""
CMD+=" --run_name \"$RUN_NAME\""
CMD+=" --output_name \"$OUTPUT_NAME\""
CMD+=" --threshold_loss $THRESHOLD_LOSS"
CMD+=" --min_steps $MIN_STEPS"
CMD+=" --lr_start $LR_START"
CMD+=" --transition_steps $TRANSITION_STEPS"
CMD+=" --decay_rate $DECAY_RATE"
CMD+=" --global_norm_clip $GLOBAL_NORM_CLIP"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --scaling $SCALING"
CMD+=" --max_dist $MAX_DIST"
CMD+=" --min_dist $MIN_DIST"

echo "Running analysis script with the following configuration:"
echo "  Input Folder: $FOLDER_NAME"
echo "  Run Name: $RUN_NAME"
echo "  Output File: $OUTPUT_NAME"
echo "-----------------------------------------------------"
echo "Executing command:"
echo "$CMD" 
echo "-----------------------------------------------------"

eval $CMD 

echo "-----------------------------------------------------"
echo "Analysis script finished."
