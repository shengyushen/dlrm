#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_pt_bin="python -u dlrm_s_pytorch.py"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)

# the same table size of criteo
#--arch-embedding-size="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"
for bs in 256 1024 4096 16384 65536
do 
  for arch_sparse_feature_size in 128 
  do

    # GPU version
    #CUDA_VISIBLE_DEVICES=0 nvprof --csv --print-api-trace  --print-gpu-trace --print-nvlink-topology --print-pci-topology \
    #  -o "criteo_%h_%p.prof" \
    CUDA_VISIBLE_DEVICES=0 \
    $dlrm_pt_bin --arch-sparse-feature-size=128 --arch-embedding-size="1460-583-1013122-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572" --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --raw-data-file=./input/day --processed-data-file=./input/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=${bs} --print-freq=128 --print-time --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --nepochs=128 --use-gpu $dlrm_extra_option  2>&1 | tee run_kaggle_pt.log_bs${bs}_arch128_gpu
  
    # CPU version
    $dlrm_pt_bin --arch-sparse-feature-size=128 --arch-embedding-size="1460-583-1013122-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572" --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --raw-data-file=./input/day --processed-data-file=./input/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=${bs} --print-freq=128 --print-time --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --nepochs=128 $dlrm_extra_option            2>&1 | tee run_kaggle_pt.log_bs${bs}_arch128
  done

done

# this is the old version
#$dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log

echo "done"
