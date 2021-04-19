#!/usr/bin/env bash


# TODO: set `data_dir` and `device`

set -exu  # x: stack trace
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

export DISABLE_TQDM=True

# options
datasplit="test"

dataset="webtext"
model_size=${1}
model_name="gpt2-${model_size}"

device=${2}

if [ ${model_name} == "gpt2-small" ]; then
    model_name="gpt2"
fi


# Default args
if [ ${dataset} == "webtext" ]; then
    data_dir="./data" # TODO
else
    data_dir="UNKNOWN dataset ${dataset}"
    exit 100
fi
args="  --data_dir ${data_dir} --model_name ${model_name} --device ${device} "

for generate_seed in 0 1 2 3 4
do

options="--datasplit ${datasplit} ${args}"
options="${options} --generate_seed ${generate_seed} --seed 1234"

##################
# basic
##################
# nucleus
for p in 0.9 0.92 0.95 0.99 1.0
do
    time python -u compute_all_L_metrics.py ${options} --generation_type basic --top_p ${p} > outs/basic/all_p_${p}_${generate_seed}_${model_size} 2>&1
done

# top-k
for k in 1
do
    time python -u compute_all_L_metrics.py ${options} --generation_type basic --top_k ${k} > outs/basic/all_k_${k}_${generate_seed}_${model_size} 2>&1
done

done # seed

############ DONE ###########

date
