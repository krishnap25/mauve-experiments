#!/bin/bash
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
set -ex

# try these values of lambda
lambdas="0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0"

dout=4           # output dimension
db="valid_feats"         # use this dataset
quant=zn_50       # cross-validate using this quantizer
best_lambda=-1
best_perf="0.0000"

for lambda in $lambdas; do
    mkdir -p test_ckpt/$lambda
    time python -u train.py \
      --dout $dout \
      --database $db \
      --lambda_uniform $lambda \
      --checkpoint_dir test_ckpt/$lambda \
      > >(tee -a test_ckpt/$lambda.stdout) 2> >(tee -a test_ckpt/$lambda.log)

    # extract validation accuracy
    perf=$(tac test_ckpt/$lambda.stdout |
                  grep -m1 'keeping as best' |
                  grep -o '(.*<' | grep -o '[0-9\.]*')

    echo $perf

    if [[ "$perf" < "$best_perf" ]]; then
        best_perf=$perf
        best_lambda=$lambda
    fi
done

echo "Best value of lambda: $best_lambda"

python eval.py \
       --database $db \
       --quantizer $quant \
       --ckpt-path test_ckpt/$best_lambda/checkpoint.pth.best
