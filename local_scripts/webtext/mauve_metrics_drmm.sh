#!/usr/bin/env bash

# TODO: set `njobs` (number of jobs to run at one) and `data_dir`

source local_scripts/parallelize.sh
njobs=32 # TODO
cmds=""

set -u  # x: stack trace
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=""
export DISABLE_TQDM=True

# options
datasplit="test"
dataset="webtext"
model_name="gpt2-large"
max_len=${1}

# Default args
if [ ${dataset} == "webtext" ]; then
    data_dir="./data" #TODO
else
    data_dir="UNKNOWN dataset ${dataset}"
    exit 100
fi
args="  --data_dir ${data_dir} --model_name ${model_name} "

for generate_seed in 0 1 2 3 4
do
for discretization in "drmm"
do

options=" ${args} --datasplit ${datasplit} --discretization ${discretization} --device -1"
options="${options} --drmm_num_epochs 20 --drmm_n_layer 3 --drmm_n_component_per_layer 10"
options="${options} --generate_seed ${generate_seed} --seed 1234"
options="${options} --use_large_feats --max_len ${max_len}"

##################
# basic
##################
# nucleus
for p in 0.8 0.9 0.92 0.95 0.99
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_p ${p} > outs/basic/mauve_${discretization}_p_${p}_${generate_seed} 2>&1 "
done

# top-k
for k in 1 5 10 50 100 500 1000 2000 5000
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_k ${k} > outs/basic/mauve_${discretization}_k_${k}_${generate_seed} 2>&1 "
done

# temperature
for t in 0.7 0.8 0.9 0.95 1.0
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --temp ${t} > outs/basic/mauve_${discretization}_t_${t}_${generate_seed} 2>&1 "
done


# top-k + temp
for t in 0.75 0.9
do
for k in 10 100
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_k ${k} --temp ${t} > outs/basic/mauve_${discretization}_k_${k}_t_${t}_${generate_seed} 2>&1 "
done
done




done # discretization
done # seed

############ DONE ###########

echo "executing..."
date
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"
date
