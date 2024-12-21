#!/bin/bash

# Configuration
DATASETS=("dataset1" "dataset2")
LANGUAGES=("en" "de")
MODEL_NAME="bert-base-multilingual-cased"
OUTPUT_DIR="./fine_tune_output"

# Loop through random seeds
for SEED in {1..10}; do
    echo "Running with random seed $SEED..."

    python main.py \
        --do_train \
        --do_test \
        --run_name "baseline_finetuner" \
        --datasets "${DATASETS[@]}" \
        --languages "${LANGUAGES[@]}" \
        --model_name_or_path "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 \
        --learning_rate 5e-5 \
        --num_epochs 3 \
        --fp16 \
        --seed "$SEED"
done
