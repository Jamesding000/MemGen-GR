#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    main.py \
    --model=TIGER \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors