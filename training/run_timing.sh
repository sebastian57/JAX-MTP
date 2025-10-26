#!/bin/bash

set -e 
PYTHON_EXE="python3" 
SCRIPT_PATH="./motep_jax_train_timing.py" 

THRESHOLD_LOSS=1e-7
MIN_STEPS=10
LR_START=3e-1
TRANSITION_STEPS=50
DECAY_RATE=1.0
GLOBAL_NORM_CLIP=9.9e-01
BATCH_SIZE=5
SCALING=1.0
MAX_DIST=5.0
MIN_DIST=0.5
TRAINING_CFG="training.cfg" 
MIN_LEV=2
MAX_LEV=24
STEPS1=1400
STEPS2=2000  
STEPS3=0  
OUTPUT_NAME=""
FOLDER_NAME=""
TRAIN="true"
TRAIN_MINIBATCH="false"
MEMORY="false"
SAVE="false"
PLOT="true"
PKL_FILE="jax_images_data"
PKL_FILE_VAL="val_jax_images_data"

CMD="$PYTHON_EXE $SCRIPT_PATH"

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
CMD+=" --training_cfg \"$TRAINING_CFG\""
CMD+=" --min_lev $MIN_LEV"
CMD+=" --max_lev $MAX_LEV"
CMD+=" --steps1 $STEPS1"
CMD+=" --steps2 $STEPS2"
CMD+=" --steps3 $STEPS3"
CMD+=" --name \"$OUTPUT_NAME\""
CMD+=" --folder_name \"$FOLDER_NAME\""
CMD+=" --train $TRAIN"
CMD+=" --train_minibatch $TRAIN_MINIBATCH"
CMD+=" --memory $MEMORY"
CMD+=" --save $SAVE"
CMD+=" --plot $PLOT"
CMD+=" --pkl_file $PKL_FILE"
CMD+=" --pkl_file_val $PKL_FILE_VAL"


echo "Running timing script with the following configuration:"
echo " Scaling: $SCALING, Min dist: $MIN_DIST, Max dist: $MAX_DIST"
echo "  Loss threshold: $THRESHOLD_LOSS, Minimum steps: $MIN_STEPS, Initial learning rate: $LR_START"
echo "  Learning rate transition steps: $TRANSITION_STEPS, Decay rate: $DECAY_RATE, Global clipping norm: $GLOBAL_NORM_CLIP"
echo "  Batch size: $BATCH_SIZE"
echo "  Config File: $TRAINING_CFG"
echo "  Training Function: Train1: $TRAIN, Train2: $TRAIN_MINIBATCH, Memory: $MEMORY"
echo "  Levels:      $MIN_LEV to $MAX_LEV (exclusive)"
echo "  Steps1:      $STEPS1"
echo "  Steps2:      $STEPS2" 
echo "  Steps3:      $STEPS3" 
echo "  Output Name: $OUTPUT_NAME"
echo "  Folder Name: $FOLDER_NAME"
echo " Save: $SAVE"
echo " Plot: $PLOT"
echo "-----------------------------------------------------"
echo "Executing command:"
echo "$CMD" 
echo "-----------------------------------------------------"

eval $CMD 

echo "-----------------------------------------------------"
echo "Timing script finished."
