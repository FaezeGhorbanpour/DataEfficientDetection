#!/bin/bash
BASE="/mounts/work/faeze/data_efficient_hate"

# Configuration
DATASETS=('dyn21_en' 'fou18_en' 'ken20_en')
LANGUAGES=('en' 'en' 'en')
SEEDS=(42 30 0 100 127)

MODEL_NAME="FacebookAI/xlm-roberta-base"

for dataset in "${DATASETS[@]}"; do
    for split in 20000; do
        for rs in rs1 rs2 rs3 rs4 rs5; do
            OUTPUT_DIR = "${BASE}/models/finetuner/roberta-default/${dataset}-${split}/${rs}/"
            python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
                --datasets "[\"${dataset}-${split}-${rs}\"]" \
                --languages "[\"${LANGUAGES[@]}\"]" \
                --seed "${SEEDS[@]}" \
                --do_fine_tuning \
                --do_train \
                --do_test \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --num_train_epochs 5 \
                --max_seq_length 128 \
                --output_dir ${OUTPUT_DIR} \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name "fine_tuning_baseline"

            # Clean up checkpoint files
            rm -rf "${OUTPUT_DIR}/check*"
        done
    done
done
