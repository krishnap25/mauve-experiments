#!/usr/bin/env bash

# TODO: set `njobs` (number of jobs to run at one) and `data_dir`

source local_scripts/parallelize.sh
njobs=28  # TODO
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

options="${args} --datasplit ${datasplit} --device -1"
options="${options} --generate_seed ${generate_seed} --seed 1234"

##################
# basic
##################
# nucleus
for p in 0.8 0.9 0.92 0.95 0.99
do
    cmds="$cmds ; time python -u compute_self_bleu_metric.py ${options} --generation_type basic --top_p ${p} > outs/basic/bleu_p_${p}_${generate_seed} 2>&1 "
done

# top-k
for k in 1 5 10 50 100 500 1000 2000 5000 10000
do
    cmds="$cmds ; time python -u compute_self_bleu_metric.py ${options} --generation_type basic --top_k ${k} > outs/basic/bleu_k_${k}_${generate_seed} 2>&1 "
done

# temperature
for t in 0.7 0.8 0.9 0.95 1.0
do
    cmds="$cmds ; time python -u compute_self_bleu_metric.py ${options} --generation_type basic --temp ${t} > outs/basic/bleu_t_${t}_${generate_seed} 2>&1 "
done


# top-k + temp
for t in 0.75 0.9
do
for k in 10 100
do
    cmds="$cmds ; time python -u compute_self_bleu_metric.py ${options} --generation_type basic --top_k ${k} --temp ${t} > outs/basic/bleu_k_${k}_t_${t}_${generate_seed} 2>&1 "
done
done

##################
# beam
##################
for bs in 4 8
do
for t in 1.0 0.9
do
for nr in 0 4
do

    fn=outs/beam/bleu_b${bs}_t${t}_n${nr}_${generate_seed}
    cmds="$cmds ; time python -u compute_self_bleu_metric.py ${options} --generation_type beam --beam_size ${bs} --temp ${t} --no_repeat_ngram ${nr} > ${fn} 2>&1 "

done # nr
done # temp
done # bs

##################
# entmax
##################
for alpha in "1.2"
do
fn=outs/entmax/bleu_entmax${alpha}_${generate_seed}
cmds="$cmds ; time python -u compute_self_bleu_metric.py ${options} --generation_type entmax --entmax_alpha ${alpha} > ${fn} 2>&1 "
done


done # seed

############ DONE ###########

echo "executing..."
date
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"
date