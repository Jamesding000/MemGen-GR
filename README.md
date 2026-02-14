# On the Memorization and Generalization of Generative Recommendation

## Quick Start

Model Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=SASRec \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors
```

Dataset Statistic

```bash
CUDA_VISIBLE_DEVICES=0 python statistics.py \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors
```

Load Checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=SASRec \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors \
    --checkpoint=saved/SASRec-Amazon14-Sports.pth \
    --epochs=0
```

Get NDCG@10 on each label
```bash
CUDA_VISIBLE_DEVICES=0 python fine-grained-results.py \
    --model=SASRec \
    --dataset=AmazonReviews2014 \
    --category=Sports_and_Outdoors \
    --checkpoint=saved/SASRec-Amazon14-Sports.pth
```
