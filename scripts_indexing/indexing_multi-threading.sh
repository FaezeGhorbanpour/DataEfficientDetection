#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

# Configuration
DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr' 'de' 'tr')
SPLITS=('train')
RSS=(rs1 rs2 rs3 rs4 rs5)

# Function to process a single dataset
run_dataset() {
    local dataset=$1
    local lang=$2
    local gpu=$3

    echo "Starting dataset: ${dataset} on GPU: ${gpu}"

    for ((i=0; i<${#RSS[@]}; i++)); do
      OUTPUT_DIR="${BASE}/models/embedder/${dataset}/LaBSE-HNSW/${RSS[i]}"
      CUDA_VISIBLE_DEVICES=${gpu} python main.py \
          --finetuner_model_name_or_path "${MODEL_NAME}" \
          --finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
          --datasets "${dataset}-2000-${RSS[i]}"  \
          --languages "${lang}" \
          --seed "${RSS[i]}" \
          --do_embedding \
          --embedder_model_name_or_path "labse"\
          --do_indexing\
          --index_type "HNSW"\
          --splits ${SPLITS[@]} \
          --index_path $OUTPUT_DIR \
          --output_dir $OUTPUT_DIR \
          --add_uncertainty \
          --uncertainty_model "/mounts/data/proj/faeze/data_efficient_hate/models/finetuner/twitter-roberta-english/fou18_en-20000/checkpoint-2500"\
          --add_perplexity\
          --cache_dir "${BASE}/cache/" \
          --logging_dir "${BASE}/logs/" \
          --overwrite_output_dir \
          --wandb_run_name "embedding-${dataset}"
    done

    echo "Finished dataset: ${dataset} on GPU: ${gpu}"
}


# Minimum GPU memory required (in MiB)
MIN_MEM=30000
# Time to wait before rechecking (in seconds)
WAIT_TIME=90
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
start_gpu=0
for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    lang=${LANGUAGES[$i]}
    #$(nvidia-smi --list-gpus | wc -l) # Get the total number of GPUs
    gpu_found=false

    for ((gpu_id=start_gpu; gpu_id<num_gpus; gpu_id++)); do
        available_gpu=$(check_gpu_memory $gpu_id)

        if [ "$available_gpu" -ge 0 ]; then
            echo "GPU $available_gpu has enough memory. Starting Python script..."
            run_dataset "$dataset" "$lang" "$available_gpu" &
            start_gpu=$((available_gpu + 1))  # Use the next GPU after the one just used
            gpu_found=true
            break
        fi
    done

    if [ "$gpu_found" = false ]; then
        echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
        sleep $WAIT_TIME
        start_gpu=0  # Reset to start checking from the first GPU again
    fi

    if [ $start_gpu -ge $num_gpus ]; then
        echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
        sleep $WAIT_TIME
        start_gpu=0  # Reset to start checking from the first GPU again
    fi
done
# Wait for all background processes to finish
wait

echo "All datasets processed!"
