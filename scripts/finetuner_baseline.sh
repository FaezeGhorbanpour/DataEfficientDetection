#!/bin/bash

# Configuration
DATASETS=('bas19_es' 'for19_pt' 'has21_hi' 'ous19_ar' 'ous19_fr' 'san20_it')
#DATASETS=('dyn21_en' 'fou18_en' 'ken20_en')
LANGUAGES=("es" "pt" "hi" "ar" "fr" "it")
SEEDS=(42 30 100 0 27)
MODEL_NAME="FacebookAI/xlm-roberta-base"

for dataset in DATASETS; do
    for split in 10 20 30 40 50 100 200 300 400 500 1000 2000; do
        for rs in rs1 rs2 rs3 rs4 rs5; do
            python main.py \
                --model_name_or_path MODEL_NAME \
                --datasets "baseline_data/${dataset}-${split}-${rs}" \
                --languages "${LANGUAGES[@]}" \
                --seed "${SEEDS[@]}" \
                --do_fine_tuning \
                --do_train \
                --do_test \
                --batch_size 16 \
                --num_train_epochs 5 \
                --max_seq_length 128 \
                --output_dir $BASE/models/${basemodel}/${dataset}/${split}/${rs} \
                --overwrite_output_dir

            rm -rf $BASE/${basemodel}/${dataset}/${split}/${rs}/check*
        done
    done
done

## Loop through random seeds
#for SEED in {1..10}; do
#    python main.py \
#        --do_train \
#        --do_test \
#        --run_name "baseline_finetuner" \
#        --datasets "${DATASETS[@]}" \
#        --languages "${LANGUAGES[@]}" \
#        --model_name_or_path "$MODEL_NAME" \
#        --output_dir "$OUTPUT_DIR" \
#        --batch_size 16 \
#        --learning_rate 5e-5 \
#        --num_epochs 3 \
#        --fp16 \
#        --seed "$SEED"
#done
