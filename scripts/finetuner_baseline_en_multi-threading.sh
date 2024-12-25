#!/bin/bash
BASE="/mounts/work/faeze/data_efficient_hate"

# Configuration
DATASETS=('dyn21_en' 'fou18_en' 'ken20_en')
DATASETS=('measure_en' 'xplain_en' 'xdomain_en')
LANGUAGES=('en' 'en' 'en')
SEEDS=(42 30 0 100 127)
RSS=(rs1 rs2 rs3 rs4 rs5)
GPUS=(3 4 5) # Adjust based on available GPUs

MODEL_NAME="microsoft/mdeberta-v3-base"
MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
MODEL_NAME="FacebookAI/xlm-roberta-base"

# Function to process a single dataset
run_dataset() {
    local dataset=$1
    local lang=$2
    local gpu=$3

    echo "Starting dataset: ${dataset} on GPU: ${gpu}"

    for split in 10000 20000; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            OUTPUT_DIR="${BASE}/models/finetuner/reboerta-default/${dataset}-${split}/${RSS[i]}/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
                --datasets "${dataset}-${split}-${RSS[i]}" \
                --languages "${lang}" \
                --seed "${SEEDS[i]}" \
                --do_fine_tuning \
                --do_train \
                --do_test \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --num_train_epochs 5 \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name "fine_tuning_mono"

        done
    done

    echo "Finished dataset: ${dataset} on GPU: ${gpu}"
}

# Launch each dataset on a separate GPU
for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    lang=${LANGUAGES[$i]}
    gpu=${GPUS[$i]}

    run_dataset "${dataset}" "${lang}" "${gpu}" &
done

# Wait for all background processes to finish
wait

echo "All datasets processed!"
