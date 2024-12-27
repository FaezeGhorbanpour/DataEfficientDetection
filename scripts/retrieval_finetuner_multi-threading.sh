#!/bin/bash
BASE="/mounts/work/faeze/data_efficient_hate"

# Configuration
DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
DATASETS=('gahd24_de' 'xdomain_tr')
LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')
LANGUAGES=('de' 'tr')
RSS=(rs1 rs2 rs3 rs4 rs5)
#RSS=(rs2 rs4 rs5)
GPUS=(5 6 7) # Adjust based on available GPUs

#MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
#FOLDER_NAME="twitter-roberta"
#FOLDER_SUBNAME="default"

#MODEL_NAME="microsoft/mdeberta-v3-base"
#FOLDER_NAME="mdeberta"
#FOLDER_SUBNAME="default"

MODEL_NAME="FacebookAI/xlm-roberta-base"
FOLDER_NAME="roberta"
FOLDER_SUBNAME="default"

# Function to process a single dataset
run_dataset() {
    local dataset=$1
    local lang=$2
    local gpu=$3

    echo "Starting dataset: ${dataset} on GPU: ${gpu}"

    for split in 20 40 1000 2000; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            OUTPUT_DIR="${BASE}/models/finetuner/${FOLDER_NAME}-${FOLDER_SUBNAME}/${dataset}-${split}/${RSS[i]}/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		--finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
                --datasets "${dataset}-${split}-${RSS[i]}" \
                --languages "${lang}" \
                --seed ${RSS[i]//rs/} \
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
            rm -rf "${OUTPUT_DIR}checkpoint*"
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
