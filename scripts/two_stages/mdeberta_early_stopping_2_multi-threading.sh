#!/bin/bash
BASE="/mounts/data/proj/faeze/transferability_hate"

# Configuration
DATASETS=("bas19_es" 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr' "dyn21_en" "fou18_en" "ken20_en" "xplain_en" "implicit_en" "xdomain_en")
#DATASETS=("xdomain_en" "implicit_en" "bas19_es" "gahd24_de" "xdomain_tr")
LANGUAGES=("es" 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr' "en" "en" "en" "en" "en" "en")
#LANGUAGES=("en" "en" "es" "de" "tr")
RSS=(rs1 rs2 rs3 rs4 rs5)

MODEL_NAME="microsoft/mdeberta-v3-base"
FOLDER_NAME="mdeberta-early-stopping-2"

# Function to process a single dataset
run_dataset() {
    local first_dataset=$1
    local first_language=$2
    local gpu=$3


    echo "Starting dataset: ${first_dataset} on GPU: ${gpu}"

    for ((d=0; d<${#DATASETS[@]}; d++)); do
      dataset=${DATASETS[d]}
      language=${LANGUAGES[d]}
      for split in 2000; do
          for ((i=0; i<${#RSS[@]}; i++)); do
              FIRST_OUTPUT_DIR="${BASE}/results/${FOLDER_NAME}/first/${first_dataset}/${split}/${RSS[i]}/"
              SECOND_OUTPUT_DIR="${BASE}/results/${FOLDER_NAME}/second/${dataset}/${first_dataset}/${split}/${RSS[i]}/"
              CUDA_VISIBLE_DEVICES=${gpu} python second_main.py \
                  --seed ${RSS[i]//rs/} \
                  --num_train_epochs 5 \
                  --do_first_fine_tuning\
                  --first_datasets "${first_dataset}-${split}-${RSS[i]}"\
                  --first_languages "${first_language}"\
                  --do_train\
                  --do_eval\
                  --do_test\
                  --do_hate_check\
                  --do_hate_day\
                  --output_dir "${FIRST_OUTPUT_DIR}" \
                  --do_second_fine_tuning\
		              --do_second_early_stopping\
		              --second_num_train_epochs 50\
                  --second_datasets "${dataset}-${split}-${RSS[i]}"\
                  --second_languages "${language}"\
                  --do_second_train\
                  --do_second_eval\
                  --do_second_test\
                  --do_second_hate_check\
                  --do_second_hate_day\
                  --second_output_dir "${SECOND_OUTPUT_DIR}" \
                  --finetuner_model_name_or_path "${MODEL_NAME}" \
                  --finetuner_tokenizer_name_or_path "${MODEL_NAME}"\
                  --per_device_train_batch_size 8 \
                  --per_device_eval_batch_size 8 \
                  --gradient_accumulation_steps 4\
                  --max_seq_length 200 \
                  --cache_dir "${BASE}/cache/" \
                  --logging_dir "${BASE}/logs/" \
                  --overwrite_output_dir \
                  --report_to None\
                  --wandb_run_name "${FOLDER_NAME}-${first_dataset}-${dataset}"

              for dir in "${OUTPUT_DIR}"check*; do
                  if [ -d "$dir" ]; then # Check if it's a directory
                      rm -rf "$dir"
                      echo "Deleted: $dir"
                  fi
              done
              for dir in "${SECOND_OUTPUT_DIR}"check*; do
                  if [ -d "$dir" ]; then # Check if it's a directory
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
# Time to wait before rechecking (in seconds)
WAIT_TIME=35000

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
D=0
while [ "$D" -lt "${#DATASETS[@]}" ]; do
    num_gpus=8
#$(nvidia-smi --list-gpus | wc -l) # Get the total number of GPUs

    for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
        available_gpu=$(check_gpu_memory $gpu_id)

        if [ "$available_gpu" -ge 0 ]; then
            echo "GPU $available_gpu has enough memory. Starting Python script..."
            run_dataset "${DATASETS[$D]}" "${LANGUAGES[$D]}" "$available_gpu" &
            
            D=$((D + 1)) # Increment D only when a GPU is assigned
            if [ "$D" -ge "${#DATASETS[@]}" ]; then
                break # Exit the loop when all datasets have been processed
            fi
        fi
    done

    echo "Reached the end of GPUs. Waiting for $WAIT_TIME seconds..."
    sleep $WAIT_TIME

done

wait

echo "All FIRST DATASETS processed!"
