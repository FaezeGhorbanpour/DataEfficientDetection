#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

# Configuration
#DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
#LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')
RSS=(rs1 rs2 rs3 rs4 rs5)

MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="mmr"

#MODEL_NAME="microsoft/mdeberta-v3-base"
#FOLDER_NAME="mdeberta"

#MODEL_NAME="FacebookAI/xlm-roberta-base"
#FOLDER_NAME="roberta"

KS=(20000 10000 5000 4000 3000 2000 1000 500 400 300 200 100 50 40 30 20 10)
KS=(10000 5000 4000 3000 1000 500 400 300 100 50 40 30)
#KS=(20000 2000 200 20)
# 20000)
# Function to process a single dataset
run_dataset() {
    local k=$1
    local gpu=$2

    # Determine epoch based on k
    local epoch
    if [ "$k" -lt 9999 ]; then
        epoch=10
    else
        epoch=5
    fi

    dataset="gahd24_de"
    lang="de"
    excluded_datasets=("gahd24_de" "dyn21_en")

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
                --index_path "/mounts/data/proj/faeze/data_efficient_hate/models/retriever/all_multilingual_with_m3/" \
                --num_retrieved ${k} \
                --exclude_datasets ${excluded_datasets[@]} \
                --combine_train_set\
                --mmr_threshold 0.99\
                --num_train_epochs ${epoch} \
                --do_fine_tuning\
                --do_train\
                --do_eval\
                --do_test\
                --do_hate_check\
                --do_hate_day\
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		--finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --report_to None\
                --wandb_run_name ${FOLDER_NAME}

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


# Minimum GPU memory required (in MiB)
MIN_MEM=8000
# Time to wait before rechecking (in seconds)
WAIT_TIME=30000

# Function to check available memory on a GPU
check_gpu_memory() {
    local gpu_id=$1
    local available_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id)

    if [ "$available_mem" -ge "$MIN_MEM" ]; then
        echo $gpu_id
    else
        echo -1
    fi
}

# Main loop
K=0
while [ "$K" -lt "${#KS[@]}" ]; do
    num_gpus=4
#$(nvidia-smi --list-gpus | wc -l) # Get the total number of GPUs

    for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
        available_gpu=$(check_gpu_memory $gpu_id)

        if [ "$available_gpu" -ge 0 ]; then
            echo "GPU $available_gpu has enough memory. Starting Python script..."
            run_dataset "${KS[$K]}" "$available_gpu" &
            sleep 30
            K=$((K + 1)) # Increment K only when a GPU is assigned
            if [ "$K" -ge "${#KS[@]}" ]; then
                break # Exit the loop when all datasets have been processed
            fi
        fi
    done

    echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
    sleep $WAIT_TIME

done

wait

echo "All K processed!"
