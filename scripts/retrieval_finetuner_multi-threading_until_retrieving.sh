#!/bin/bash
BASE="/mounts/work/faeze/data_efficient_hate"

# Configuration
#DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
#LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')
RSS=(rs1 rs2 rs3 rs4 rs5)
GPUS=(0 1 2 3 4 5 6 7) # Adjust based on available GPUs

MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="twitter-roberta"

#MODEL_NAME="microsoft/mdeberta-v3-base"
#FOLDER_NAME="mdeberta"

#MODEL_NAME="FacebookAI/xlm-roberta-base"
#FOLDER_NAME="roberta"

#KS=(20 30 40 50 100 200 300 400)
KS=(500 1000 2000 3000 4000 5000 10000 20000)

# Function to process a single dataset
run_dataset() {
    local k=$1
    local gpu=$2

    # Determine epoch based on k
    local epoch
    if [ "$k" -lt 10000 ]; then
        epoch=5
    else
        epoch=3
    fi

    dataset="ous19_fr"
    lang="fr"

    echo "Starting k: ${k} on GPU: ${gpu}"

    for split in 10 20 30 40 50 100 200 300 400 500 1000 2000; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            OUTPUT_DIR="${BASE}/models/retrieval_finetuner/${FOLDER_NAME}/${dataset}/${split}/${k}/${RSS[i]}/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --datasets "${dataset}-${split}-${RSS[i]}" \
                --languages "${lang}" \
                --seed ${RSS[i]//rs/} \
                --do_embedding \
                --embedder_model_name_or_path "m3" \
                --do_searching \
                --splits "train" \
                --index_path "/mounts/work/faeze/data_efficient_hate/models/retriever/all_multilingual_with_m3/" \
                --max_retrieved ${k} \
                --exclude_datasets "\[${dataset}\]" \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name "retrieval_finetuning"

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


wait_for_free_gpu() {
    local required_memory=5000 # Memory in MB
    while true; do
        for gpu in "${GPUS[@]}"; do
            free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk "NR==${gpu}+1")
            if [ "$free_memory" -gt "$required_memory" ]; then
                echo "GPU $gpu has enough free memory: ${free_memory}MB"
                echo $gpu
                return
            fi
        done
        echo "Waiting for available GPU..."
        sleep 30
    done
}

# Main Execution Loop
for k in "${KS[@]}"; do
    gpu=$(wait_for_free_gpu) # Wait for a GPU with sufficient free memory
    run_dataset "${k}" "${gpu}"
done

echo "All K processed!"
