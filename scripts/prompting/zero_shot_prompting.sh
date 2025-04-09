#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <prompter_model_name_or_path> <gpu_core_number>"
    exit 1
fi


DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it' 'gahd24_de' 'xdomain_tr')
LANGUAGES=('es' 'pt' 'hi' 'ar' 'fr' 'it' 'de' 'tr')

# Get arguments
MODEL_PATH=$1
GPU_CORE=$2

echo "starting  ${MODEL_PATH} on gpu ${GPU_CORE}"

# Run the Python script with only the model path argument
CUDA_VISIBLE_DEVICES=$GPU_CORE python main.py \
    --prompter_model_name_or_path "$MODEL_PATH" \
    --num_rounds 3 \
    --datasets ${DATASETS[@]} \
    --languages ${LANGUAGES[@]} \
    --do_prompting \
    --prompter_output_dir "/mounts/data/proj/faeze/data_efficient_hate/models/prompter/zero_shot_prompting/" \
    --output_dir "/mounts/data/proj/faeze/data_efficient_hate/models/prompter/zero_shot_prompting/" \
    --cache_dir "/mounts/data/proj/faeze/data_efficient_hate/cache/" \
    --logging_dir "/mounts/data/proj/faeze/data_efficient_hate/logs/" \
    --overwrite_output_dir \
    --wandb_run_name "prompter_zero_shot: ${MODEL_PATH}"
