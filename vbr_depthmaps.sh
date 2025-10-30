#!/bin/bash

# --- Shared dataset root and pairs base path ---
VBR_ROOT="/datasets/vbr_slam/"
PAIRS_BASE_PATH="/home/bjangley/VPR/mast3r-v2/pairs_finetuning2"

# --- Base output directory ---
OUTPUT_BASE_DIR="/home/bjangley/VPR/zoedepth_vbr/"

# --- List of scenes to process ---
SCENES=("campus_train1") #"spagna_train0" "ciampino_train0" "ciampino_train1" "campus_train0" 

# --- Model and Inference Settings ---
# ZoeDepth does not take an encoder argument, so omit it
SAVE_TYPE="depth"  # or "3d"

# --- Device config ---
export CUDA_VISIBLE_DEVICES=5

for SCENE_NAME in "${SCENES[@]}"
do
    echo "Processing scene: $SCENE_NAME"
    PAIRS_FILE="$PAIRS_BASE_PATH/$SCENE_NAME/all_pairs.txt"
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$SCENE_NAME"
    mkdir -p "$OUTPUT_DIR"

    CMD=(python prepare_vbr_depthmaps.py
        --vbr_scene "$SCENE_NAME"
        --vbr_root "$VBR_ROOT"
        --pairs_file "$PAIRS_FILE"
        --output_dir "$OUTPUT_DIR"
        --save "$SAVE_TYPE"
    )

    # Execute the assembled command safely
    "${CMD[@]}"

    echo "Finished processing $SCENE_NAME"
done
