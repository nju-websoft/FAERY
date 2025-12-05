#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run_hhgt.py --dataset google --len-seq 20 --epoch 10 --patience 3 --num-heads 1 --lr 1e-4 --dropout 0.1 --beta 0.5 --temperature 2 --feats-type 0 --batch-size 256 --eval-steps 100 --top-k 20 --repeat 2 --mode rerank_union --hidden-dim 256 --fold Annotators_split --max-k 1

