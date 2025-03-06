#!/bin/bash
BASE="/mounts/data/proj/faeze/data_efficient_hate"

# Configuration
RSS=(rs1 rs2 rs3 rs4 rs5)
MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
FOLDER_NAME="best-criteria-ratio"
KS=(20 200)
CRITERIA_WEIGHTS=(0.4 0.3 0.2 0.1 0)

# Function to process a single dataset
run_dataset() {
    local k=$1
    local gpu=$2
    local unique_weight=$3

    # Determine epoch based on k
    local epoch
    if [ "$k" -lt 9999 ]; then
        epoch=10
    else
        epoch=5
    fi

    dataset="bas19_es"
    lang="es"
    excluded_datasets=("bas19_es")

    for cluster_weight in "${CRITERIA_WEIGHTS[@]}"; do
      echo "Starting k: ${k}, unique_word_criteria_weight: ${unique_weight}, cluster_criteria_weight: ${cluster_weight} on GPU: ${gpu}"
      for split in 10 20 30 40 50; do
          for ((i=0; i<${#RSS[@]}; i++)); do
              OUTPUT_DIR="${BASE}/models/retrieval_finetuner/${FOLDER_NAME}/${dataset}/${split}/${k}/${RSS[i]}/uw_${unique_weight}_cw_${cluster_weight}/"
              CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                  --datasets "${dataset}-${split}-${RSS[i]}" \
                  --languages "${lang}" \
                  --seed ${RSS[i]//rs/} \
                  --do_embedding \
                  --embedder_model_name_or_path "m3" \
                  --do_searching \
                  --balance_labels \
                  --unique_word_criteria_weight ${unique_weight} \
                  --cluster_criteria_weight ${cluster_weight} \
                  --splits "train" \
                  --index_path "/mounts/data/proj/faeze/data_efficient_hate/models/retriever/all_multilingual_with_m3/" \
                  --num_retrieved ${k} \
                  --exclude_datasets ${excluded_datasets[@]} \
                  --combine_train_set \
                  --do_fine_tuning \
                  --num_train_epochs ${epoch} \
                  --do_train \
                  --do_eval \
                  --do_test \
                  --do_hate_check \
                  --finetuner_model_name_or_path "${MODEL_NAME}" \
                  --finetuner_tokenizer_name_or_path "${MODEL_NAME}" \
                  --per_device_train_batch_size 16 \
                  --per_device_eval_batch_size 64 \
                  --max_seq_length 128 \
                  --output_dir $OUTPUT_DIR \
                  --cache_dir "${BASE}/cache/" \
                  --logging_dir "${BASE}/logs/" \
                  --overwrite_output_dir \
                  --report_to None \
                  --wandb_run_name ${FOLDER_NAME}
  
              for dir in "${OUTPUT_DIR}"check*; do
                  if [ -d "$dir" ]; then
                      rm -rf "$dir"
                      echo "Deleted: $dir"
                  fi
              done
          done
      done
    done

    echo "Finished dataset: ${dataset} on GPU: ${gpu}"
}

# Minimum GPU memory required (in MiB)
MIN_MEM=8000
WAIT_TIME=10  # Time to wait before rechecking (in seconds)

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
num_gpus=2
done_processing=0

while [ "$K" -lt "${#KS[@]}" ] && [ "$done_processing" -eq 0 ]; do
    U=0
    while [ "$U" -lt "${#CRITERIA_WEIGHTS[@]}" ]  && [ "$done_processing" -eq 0 ]; do
        for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
            available_gpu=$(check_gpu_memory $gpu_id)
            if [ "$available_gpu" -ge 0 ]; then
                echo "GPU $available_gpu assigned to K=${KS[$K]}, uw=${CRITERIA_WEIGHTS[$U]}"

                # Iterate over all C values inside the same GPU process (sequentially)
                # echo "Running K=${KS[$K]}, uw=${CRITERIA_WEIGHTS[$U]}"
                run_dataset "${KS[$K]}" "$available_gpu" "${CRITERIA_WEIGHTS[$U]}" &

                U=$((U + 1))
                if [ "$U" -ge "${#CRITERIA_WEIGHTS[@]}" ]; then
                        U=0  # Reset U and move to the next cluster_criteria_weight (C)
                        K=$((K + 1))
                        if [ "$K" -ge "${#KS[@]}" ]; then
                           done_processing=1  # Signal all loops to exit
                        fi
                fi
            fi

                if [ "$done_processing" -eq 1 ]; then
                    break
                fi
        done
    done
    echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
    sleep $WAIT_TIME
done

wait

echo "All K and criteria weight combinations processed!"

