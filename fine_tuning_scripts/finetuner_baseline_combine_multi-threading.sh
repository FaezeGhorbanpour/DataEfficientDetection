#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

# Configuration
#DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it')
DATASETS=('gahd24_de' 'xdomain_tr')
DATASETS=('bas19_es')
#LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')
LANGUAGES=('de' 'tr')
LANGUAGES=('es')
RSS=(rs1 rs2 rs3 rs4 rs5)

MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="twitter-roberta"
FOLDER_SUBNAME="combine_dyn21_with_shuffling"

# Function to process a single dataset
run_dataset() {
    local dataset=$1
    local lang=$2
    local gpu=$3


    echo "Starting dataset: ${dataset} on GPU: ${gpu}"

    for split in 2000 1000 500 400 300 200 100 50 40 30 20 10; do
        for ((i=0; i<${#RSS[@]}; i++)); do
            datas=("dyn21_en-20000-${RSS[i]}" "${dataset}-${split}-${RSS[i]}")
            langs=("en" "${lang}")
            OUTPUT_DIR="${BASE}/models/finetuner/${FOLDER_NAME}-${FOLDER_SUBNAME}/${dataset}/${split}/${RSS[i]}/"
            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		--finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
                --datasets "${datas[@]}" \
                --languages "${langs[@]}" \
                --seed ${RSS[i]//rs/} \
                --combine_train_set\
                --do_fine_tuning \
                --do_train \
                --do_test \
		--shuffle \
                --do_hate_check\
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --num_train_epochs 3 \
                --max_seq_length 128 \
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
MIN_MEM=7000
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
start_gpu=7
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
