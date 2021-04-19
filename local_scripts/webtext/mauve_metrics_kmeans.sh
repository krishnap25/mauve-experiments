#!/usr/bin/env bash

# TODO: set `njobs` (number of jobs to run at one) and `data_dir`

source local_scripts/parallelize.sh
njobs=24 # TODO
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
#model_name="gpt2-large"
#max_len=${1}

# Default args
if [ ${dataset} == "webtext" ]; then
    data_dir="./data"  # TODO
else
    data_dir="UNKNOWN dataset ${dataset}"
    exit 100
fi

discretization="kmeans_l2"
kmeans_num_clusters=500

for max_len in 1024 512 256 128
do
for generate_seed in 0 1 2 3 4
do
for model_name in "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl"
do

args="  --data_dir ${data_dir} --model_name ${model_name} "
sn="${discretization}_${kmeans_num_clusters}_${model_name}_${max_len}"

options="${args} --datasplit ${datasplit} --discretization ${discretization} --device -1"
options="${options} --kmeans_num_clusters ${kmeans_num_clusters}"
options="${options} --generate_seed ${generate_seed} --seed 1234"
options="${options} --use_large_feats --max_len ${max_len} --kmeans_explained_var 0.9"

##################
# basic
##################
# nucleus
for p in 0.9 0.92 0.95 0.99
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_p ${p} > outs/basic/mauve_${sn}_p_${p}_seed${generate_seed} 2>&1 "
done

# top-k
for k in 1
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --top_k ${k} > outs/basic/mauve_${sn}_k_${k}_seed${generate_seed} 2>&1 "
done

# temperature
for t in 1.0
do
    cmds="$cmds ; time python -u compute_mauve_metrics.py ${options} --generation_type basic --temp ${t} > outs/basic/mauve_${sn}_t_${t}_seed${generate_seed} 2>&1 "
done

done # model_name
done # seed
done # length

############ DONE ###########

echo "executing..."
date
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"
date
