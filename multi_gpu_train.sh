#!/usr/bin/env bash
OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NGPUS=2
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train.py --config_file "config/bill2020_MobileNetV3_FPN_DBhead_polyLR.yaml"