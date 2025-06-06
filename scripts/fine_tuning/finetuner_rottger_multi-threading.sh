#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

# Configuration
DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'xdomain_tr' 'gahd24_de')
LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'tr' 'de')
RSS=(rs1 rs2 rs3 rs4 rs5)
GPUS=(0 0 0 0 0 0 1 1) # Adjust based on available GPUs

TOKENIZER_NAME="cardiffnlp/twitter-xlm-roberta-base"
#TOKENIZER_NAME="microsoft/mdeberta-v3-base"
MODEL_NAME="/mounts/data/proj/faeze/data_efficient_hate/models/finetuner/twitter-roberta-english/ken20_en-20000/checkpoint-3750"
FOLDER_NAME="twitter-roberta"
FOLDER_SUBNAME="ken20"
# Function to process a single dataset
run_dataset() {
    local dataset=$1
    local lang=$2
    local gpu=$3

    echo "Starting dataset: ${dataset} on GPU: ${gpu}"

    for split in 2000 1000 500 400 300 200 100 50 40 30 20 10; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            OUTPUT_DIR="${BASE}/models/finetuner/${FOLDER_NAME}-${FOLDER_SUBNAME}/${dataset}-${split}/${RSS[i]}/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		--finetuner_tokenizer_name_or_path "${TOKENIZER_NAME}" \
                --datasets "${dataset}-${split}-${RSS[i]}" \
                --languages "${lang}" \
                --seed ${RSS[i]//rs/} \
                --do_fine_tuning \
                --do_train \
                --do_test \
                --do_hate_check\
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --num_train_epochs 5 \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name "${FOLDER_SUBNAME}"

            # Clean up checkpoint files
            for dir in "${OUTPUT_DIR}"check*; do
                if [ -d "$dir" ]; then # Check if it's a directory
                    rm -rf "$dir"
                    echo "Deleted: $dir"
                fi
            done
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
