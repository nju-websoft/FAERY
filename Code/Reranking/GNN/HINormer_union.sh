#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py --dataset google --len-seq 10 --epoch 10 --patience 3 --num-gnns 1 --num-layers 2 --num-heads 1 --lr 1e-4 --dropout 0.1 --beta 0.1 --temperature 2 --feats-type 0 --batch-size 256 --eval-steps 100 --top-k 20 --repeat 1 --mode rerank_union --hidden-dim 256 --fold Annotators_split

