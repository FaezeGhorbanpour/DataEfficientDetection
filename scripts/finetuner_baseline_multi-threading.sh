#!/bin/bash

BASE="/mounts/work/faeze/data_efficient_hate"

# Configuration
DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it')
LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it')
RSS=(rs1 rs2 rs3 rs4 rs5)
SEEDS=(42 30 0 100 127)

MODEL_NAME="FacebookAI/xlm-roberta-base"

# Ensure GPUs are assigned in a round-robin fashion
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)  # Count available GPUs
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs detected. Exiting."
    exit 1
fi

run_dataset() {
    local dataset=$1
    local gpu_id=$2

    for split in 10 20 30 40 50 100 200 300 400 500 1000 2000; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            OUTPUT_DIR="${BASE}/models/finetuner/roberta-default/${dataset}-${split}/${RSS[i]}/"
            python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
                --datasets "${dataset}-${split}-${RSS[i]}" \
                --languages "${LANGUAGES[@]}" \
                --seed "${SEEDS[i]}" \
                --do_fine_tuning \
                --do_train \
                --do_test \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --num_train_epochs 10 \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name "fine_tuning_mono"

            # Clean up checkpoint files
            rm -rf "${OUTPUT_DIR}/check*"
        done
    done
}

# Launch each dataset on a separate GPU
for i in "${!DATASETS[@]}"; do
    gpu_id=$((i % NUM_GPUS))  # Assign GPU in round-robin
    dataset="${DATASETS[$i]}"
    run_dataset "$dataset" "$gpu_id" &
done

# Wait for all background processes to finish
wait

echo "All datasets processed."
