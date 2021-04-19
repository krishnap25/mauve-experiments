#!/usr/bin/env bash

# TODO: set `data_dir` and `device`
njobs=1
cmds=""


set -exu  # x: stack trace
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

export DISABLE_TQDM=True

# options
datasplit="test"
dataset="webtext"
model_name="gpt2-large"

# Default args
if [ ${dataset} == "webtext" ]; then
    data_dir="./data"  # TODO
else
    data_dir="UNKNOWN dataset ${dataset}"
    exit 100
fi
args="  --data_dir ${data_dir} --model_name ${model_name} "

for generate_seed in 0 1 2 3 4
do
discretization="spv"
device=0 # TODO
options="${args} --datasplit ${datasplit} --discretization ${discretization} --device ${device}"
options="${options} --spv_num_epochs 200 "
options="${options} --generate_seed ${generate_seed} --seed 1234"
options="${options} --use_large_feats"

##################
# basic
##################
# nucleus
for p in 0.8 0.9 0.92 0.95 0.99
do
    time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_p ${p} > outs/basic/mauve_${discretization}_p_${p} 2>&1
done

# top-k
for k in 1 5 10 50 100 500 1000 2000 5000
do
    time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_k ${k} > outs/basic/mauve_${discretization}_k_${k} 2>&1
done

# temperature
for t in 0.7 0.8 0.9 0.95 1.0
do
    time python -u compute_mauve_metrics.py ${options} --generation_type basic --temp ${t} > outs/basic/mauve_${discretization}_t_${t} 2>&1
done


# top-k + temp
for t in 0.75 0.9
do
for k in 10 100
do
    time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_k ${k} --temp ${t} > outs/basic/mauve_${discretization}_k_${k}_t_${t} 2>&1
done
done



done # seed

date
