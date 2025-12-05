#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run_hhgt.py --dataset google --len-seq 20 --epoch 10 --patience 3 --num-heads 1 --lr 1e-4 --dropout 0.1 --beta 0.1 --temperature 2 --feats-type 0 --batch-size 64 --eval-steps 100 --top-k 20 --repeat 2 --mode rerank_union --hidden-dim 256 --fold 5-Fold_split/fold_4 --max-k 1

