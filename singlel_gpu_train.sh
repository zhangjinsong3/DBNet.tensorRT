#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config_file "config/bill2020_resnet18_FPN_DBhead_polyLR.yaml"

"""
512x512 large scale=0.5 46FPS:
recall: 0.924765, precision: 0.910494, hmean: 0.917574, train_loss: 0.485246, best_model_epoch: 176.000000,

512x512 small scale=0.5 46FPS:
recall: 0.898119, precision: 0.909524, hmean: 0.903785, train_loss: 0.555229, best_model_epoch: 129.000000
"""