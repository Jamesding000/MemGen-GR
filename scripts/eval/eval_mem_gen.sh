#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

wandb_project="fine-grained-results"
log_file="logs/fine_grained_results/all_fine_grained_results_test.csv"
base_path="saved_models"

datasets=(
    "AmazonReviews2014-Sports_and_Outdoors"
    "AmazonReviews2014-Beauty"
    "AmazonReviews2023-Industrial_and_Scientific"
    "AmazonReviews2023-Musical_Instruments"
    "AmazonReviews2023-Office_Products"
    "Steam"
    "Yelp-Yelp_2020"
)

for split in test; do
    for full_name in "${datasets[@]}"; do
    
        if [[ "$full_name" == *"-"* ]]; then
            dataset="${full_name%%-*}"
            category="${full_name#*-}"
            subset_arg="--category=$category"
            ckpt_suffix="-category_${category}"
        else
            dataset="$full_name"
            subset_arg=""
            ckpt_suffix=""
        fi

        for model in SASRec TIGER; do
            echo "Running $model on $full_name..."

            checkpoint_path="${base_path}/${model}-${dataset}${ckpt_suffix}.pth"

            cmd="python mem_gen_evaluation.py \
                --model=$model \
                --dataset=$dataset \
                --eval=$split \
                --checkpoint_path=$checkpoint_path \
                --wandb_project=$wandb_project \
                --log_file=$log_file \
                --save_inference \
                $subset_arg"

            if [ "$model" == "TIGER" ]; then
                sem_ids="${base_path}/semantic_ids/${full_name}_sentence-t5-base_256,256,256,256.sem_ids"
                cmd="$cmd --sem_ids_path=$sem_ids"
            fi

            eval $cmd
        done
    done
done
