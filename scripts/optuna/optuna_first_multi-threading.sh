#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="optuna-first"

KS=(20 200 2000 20000)
# Function to process a single dataset
run_dataset() {
    local k=$1
    local gpu=$2


    dataset="bas19_es"
    lang="es"
    excluded_datasets=("bas19_es")

    echo "Starting k: ${k} on GPU: ${gpu}"

    local optuna_n_trials
    if [ "$k" -lt 9999 ]; then
        optuna_n_trials=30
    else
        optuna_n_trials=15
    fi

    for split in 20 200 2000; do
            OUTPUT_DIR="${BASE}/models/retrieval_finetuner/${FOLDER_NAME}/${dataset}/${split}/${k}/rs3/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --datasets "${dataset}-${split}-rs3" \
                --languages "${lang}" \
                --seed 3 \
                --do_embedding \
                --embedder_model_name_or_path "m3" \
                --do_searching \
                --splits "train" \
                --index_path "/mounts/data/proj/faeze/data_efficient_hate/models/retriever/all_multilingual_with_m3/" \
                --num_retrieved ${k} \
                --exclude_datasets ${excluded_datasets[@]} \
                --combine_train_set\
                --do_train\
                --do_eval\
                --run_optuna\
                --optuna_n_trials ${optuna_n_trials}\
                --optuna_study_name "${FOLDER_NAME}-${dataset}-${split}-${k}-rs3"\
                --optuna_storage_path $OUTPUT_DIR \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		            --finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --report_to None\
                --enable_wandb 0

            for dir in "${OUTPUT_DIR}"check*; do
                if [ -d "$dir" ]; then # Check if it's a directory
                    rm -rf "$dir"
                    echo "Deleted: $dir"
                fi
            done
    done

    echo "Finished dataset: ${dataset} on GPU: ${gpu}"
}


# Minimum GPU memory required (in MiB)
MIN_MEM=8000
# Time to wait before rechecking (in seconds)
WAIT_TIME=10

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
