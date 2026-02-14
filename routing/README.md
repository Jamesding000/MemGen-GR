### Memorization-aware Soft Routing

**To run the experiment pipeline end-to-end**:

1. Generate SASRec and TIGER inference results:
```
bash routing/generate_inference_results.sh
```

2. Run soft fusion hyperparameter tunning experiments:
```
bash routing/run_soft_fusion_parallel.sh
```