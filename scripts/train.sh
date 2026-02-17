#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --model=SASRec \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors

python main.py \
    --model=TIGER \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors
