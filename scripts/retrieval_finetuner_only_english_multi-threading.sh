#!/bin/bash
BASE="/mounts/work/faeze/data_efficient_hate"

# Configuration
#DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
#LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')
RSS=(rs1 rs2 rs3 rs4 rs5)
GPUS=(0 1 2 3 4 5 6 7) # Adjust based on available GPUs

MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="two_phases_only_english"

#MODEL_NAME="microsoft/mdeberta-v3-base"
#FOLDER_NAME="mdeberta"

#MODEL_NAME="FacebookAI/xlm-roberta-base"
#FOLDER_NAME="roberta"

#KS=()
KS=(500 1000 2000 3000 4000 5000 10000 20000 20 30 40 50 100 200 300 400 )

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

    dataset="bas19_es"
    lang="es"

    echo "Starting k: ${k} on GPU: ${gpu}"

    for split in 2000 1000 500 400 300 200 100 50 40 30 20 10; do
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
                --index_path "/mounts/work/faeze/data_efficient_hate/models/retriever/six_english_with_m3/" \
                --max_retrieved ${k} \
                --exclude_datasets "\[${dataset}\]" \
                --do_retrieval_tuning \
                --retrieval_num_train_epochs ${epoch} \
                --retrieval_do_train \
                --retrieval_do_test \
                --do_fine_tuning \
                --num_train_epochs 5 \
                --do_train\
                --do_eval\
                --do_test\
                --finetuner_model_name_or_path "${MODEL_NAME}" \
		--finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --cache_dir "${BASE}/cache/" \
                --logging_dir "${BASE}/logs/" \
                --overwrite_output_dir \
                --wandb_run_name $FOLDER_NAME

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
WAIT_TIME=100

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
    num_gpus=8
#$(nvidia-smi --list-gpus | wc -l) # Get the total number of GPUs

    for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
        available_gpu=$(check_gpu_memory $gpu_id)

        if [ "$available_gpu" -ge 0 ]; then
            echo "GPU $available_gpu has enough memory. Starting Python script..."
            run_dataset "${KS[$K]}" "$available_gpu" &
            #sleep 60
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
