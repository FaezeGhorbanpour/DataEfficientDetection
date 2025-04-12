#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <prompter_model_name_or_path> <gpu_core_number>"
    exit 1
fi


DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')
SPLITS=('test' 'hate_check' 'hate_day')

# Get arguments
MODEL_PATH=$1
GPU_CORE=$2

echo "starting  ${MODEL_PATH} on gpu ${GPU_CORE}"


for ((i=0; i<${#DATASETS[@]}; i++)); do
  dataset=${DATASETS[i]}
  language=${LANGUAGES[i]}
  index="/mounts/data/proj/faeze/data_efficient_hate/models/embedder/${dataset}/train/LaBSE-HNSW/"
  # Run the Python script with only the model path argument
  CUDA_VISIBLE_DEVICES=$GPU_CORE python main.py \
      --prompter_model_name_or_path "$MODEL_PATH" \
      --num_rounds 1 \
      --datasets ${dataset}\
      --languages ${language} \
      --do_embedding\
      --embedder_model_name_or_path "labse"\
      --do_searching\
      --splits ${SPLITS[@]}\
      --index_path ${index}\
      --k 50\
      --retrieve_per_instance\
      --balance_labels\
      --do_prompting \
      --prompter_output_dir "/mounts/data/proj/faeze/data_efficient_hate/models/prompter/few_shot_prompting/" \
      --output_dir "/mounts/data/proj/faeze/data_efficient_hate/models/prompter/few_shot_prompting/" \
      --cache_dir "/mounts/data/proj/faeze/data_efficient_hate/cache/" \
      --logging_dir "/mounts/data/proj/faeze/data_efficient_hate/logs/" \
      --overwrite_output_dir \
      --wandb_run_name "prompter_few_shot: ${MODEL_PATH}"

done
