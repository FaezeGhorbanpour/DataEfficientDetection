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

#KS=()
KS=(20 30 40 50 100 200 300 400 500 1000 2000 3000 4000 5000 10000 20000)

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


# Keep track of the last GPU used
LAST_GPU=-1

wait_for_free_gpu() {
    local required_memory=5000 # Memory in MB
    local total_gpus=${#GPUS[@]}

    while true; do
        # Start iterating from the next GPU after the last used one
        for ((offset=1; offset<=total_gpus; offset++)); do
            gpu_index=$(( (LAST_GPU + offset) % total_gpus ))
            gpu=${GPUS[$gpu_index]}

            # Check free memory for the current GPU
            free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk "NR==${gpu}+1")
            if [ "$free_memory" -gt "$required_memory" ]; then
                echo "GPU $gpu has enough free memory: ${free_memory}MB"
                LAST_GPU=$gpu_index
                echo $gpu
                return
            fi
        done

        # If no GPU is available, wait for 2 minutes and retry from GPU 0
        echo "No GPU available. Waiting for 2 minutes before retrying..."
        sleep 120
    done
}


# Minimum GPU memory required (in MiB)
MIN_MEM=5000
# Time to wait before rechecking (in seconds)
WAIT_TIME=200

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
for k in "${KS[@]}"; do
    num_gpus=$(nvidia-smi --list-gpus | wc -l) # Get the total number of GPUs

    for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
        available_gpu=$(check_gpu_memory $gpu_id)

        if [ "$available_gpu" -ge 0 ]; then
            echo "GPU $available_gpu has enough memory. Starting Python script..."
            run_dataset "${k}" "${available_gpu}" &
        fi
    done

    echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
    sleep $WAIT_TIME

done


echo "All K processed!"
