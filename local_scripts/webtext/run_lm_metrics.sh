#!/usr/bin/env bash
set -exu

device=1
data_dir="./data" # TODO
datasplit="test"
max_len=1024
model_name="gpt2-large"

echo "Starting at $(date)"

time python -u compute_lm_metrics_basic.py  \
    --device ${device} \
    --data_dir ${data_dir} \
    --datasplit ${datasplit} \
    --model_name ${model_name} \
    --max_len ${max_len} \
    > outs/lm_large 2>&1

echo "Done at $(date)"
