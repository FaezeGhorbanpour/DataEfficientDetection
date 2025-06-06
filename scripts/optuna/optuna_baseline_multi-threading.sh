#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

# Configuration
#DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
DATASETS=('bas19_es' 'ous19_ar')
#LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr' 'de' 'tr')
LANGUAGES=('es' 'ar')
RSS=(rs3)

MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="twitter-roberta"
FOLDER_SUBNAME="optuna"

# Function to process a single dataset
run_dataset() {
    local dataset=$1
    local lang=$2
    local gpu=$3


    echo "Starting dataset: ${dataset} on GPU: ${gpu}"

    for split in 20 200 2000; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            OUTPUT_DIR="${BASE}/models/finetuner/${FOLDER_NAME}-${FOLDER_SUBNAME}/${dataset}/${split}/${RSS[i]}/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		            --finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
		            --datasets "${dataset}-${split}-${RSS[i]}" \
                --languages "${lang}" \
                --seed ${RSS[i]//rs/} \
                --combine_train_set\
                --do_fine_tuning \
                --do_train \
                --do_test \
                --do_hate_check\
                --do_hate_day\
                --run_optuna\
                --optuna_n_trials 30\
                --optuna_study_name "${FOLDER_NAME}-${dataset}-${split}-rs3"\
                --optuna_storage_path $OUTPUT_DIR \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name "fine_tuning_combine"

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


# Minimum GPU memory required (in MiB)
MIN_MEM=10000
# Time to wait before rechecking (in seconds)
WAIT_TIME=180
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
num_gpus=8
start_gpu=1
for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    lang=${LANGUAGES[$i]}
#$(nvidia-smi --list-gpus | wc -l) # Get the total number of GPUs
    for ((gpu_id=start_gpu; gpu_id<num_gpus; gpu_id++)); do
        available_gpu=$(check_gpu_memory $gpu_id)

        if [ "$available_gpu" -ge 0 ]; then
            echo "GPU $available_gpu has enough memory. Starting Python script..."
            run_dataset "$dataset" "$lang" "$available_gpu" &
            start_gpu=$((start_gpu + 1))
            break 1
        fi
    done
    if [ $start_gpu -ge $num_gpus ]; then
        echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
        sleep $WAIT_TIME
    fi
done
# Wait for all background processes to finish
wait

echo "All datasets processed!"
