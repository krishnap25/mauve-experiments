#!/bin/bash

## TO change: `data_dir` in line 45 and output directory. Pass in model size as argument.

#SBATCH --job-name=gen_basic
#SBATCH --comment="Generate all baselines"
#SBATCH --array=0-29%4
#SBATCH --output=TODO
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL


# Initialize conda into the right environment + modules.
source ~/.bashrc
conda activate pyt17  # cuda 10.1
#conda activate pyt14_tf1  # cuda 10.0
export DISABLE_TQDM=True

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu


model_size=$1  # pass model size as argument
prompt_size=35

dataset="webtext"
model_name="gpt2-${model_size}"

if [ ${model_name} == "gpt2-small" ]; then
    model_name="gpt2"
fi

# Default args
if [ ${dataset} == "webtext" ]; then
    data_dir="./data" ###TODO
else
    data_dir="UNKNOWN dataset ${dataset}"
    exit 100
fi


list_of_jobs=()

for seed in 0 1 2 3 4
do
for datasplit in test
do

# nucleus
for p in 0.9 0.92 0.95 0.99
do
    k=0
    t=1
    job="--top_p ${p} --top_k ${k} --temp ${t} --seed ${seed}"
    list_of_jobs+=("${job}")
done

# top-k
#for k in 1 5 10 50 100 500 1000 2000 5000 1000
for k in 1
do
    p=1
    t=1
    job="--top_p ${p} --top_k ${k} --temp ${t} --seed ${seed}"
    list_of_jobs+=("${job}")
done

# temperature
#for t in 0.7 0.8 0.9 0.95 1.0
for t in 1.0
do
    p=1
    k=0
    job="--top_p ${p} --top_k ${k} --temp ${t} --seed ${seed}"
    list_of_jobs+=("${job}")
done
#
## top-k + temperature
#for t in 0.75 0.9
#do
#for k in 10 100
#do
#    p=1
#    job="--top_p ${p} --top_k ${k} --temp ${t} --seed ${seed}"
#    list_of_jobs+=("${job}")
#done
#done

done # datasplit
done # seed

num_jobs=${#list_of_jobs[@]}

job_id=${SLURM_ARRAY_TASK_ID}

if [ ${job_id} -ge ${num_jobs} ] ; then
    echo "Invalid job id; qutting"
    exit 2
fi

echo "-------- STARTING JOB ${job_id}/${num_jobs}"

args=${list_of_jobs[${job_id}]}


time python -u generate_basic.py ${args} \
    --device 0 \
    --datasplit ${datasplit} \
    --data_dir ${data_dir} \
    --model_name ${model_name} \
    --prompt_size ${prompt_size} \
    --use_large_feats

echo "Job completed at $(date)"
