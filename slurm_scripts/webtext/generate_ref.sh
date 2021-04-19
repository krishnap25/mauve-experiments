#!/bin/bash

## TODO: set output dirs (line 7) and data_dir (line 39)

#SBATCH --job-name=gen_ref
#SBATCH --comment="Generate all baselines"
#SBATCH --output=#TODO
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00


# Initialize conda into the right environment + modules.
source ~/.bashrc
conda activate pyt17  # cuda 10.1
export DISABLE_TQDM=True

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

# TODO: set dataset and model_name

model_size="large"

dataset="webtext"
model_name="gpt2-${model_size}"

if [ ${model_name} == "gpt2-small" ]; then
    model_name="gpt2"
fi

# Default args
if [ ${dataset} == "webtext" ]; then
    data_dir="./data" #TODO
else
    data_dir="UNKNOWN dataset ${dataset}"
    exit 100
fi


for datasplit in "test" "valid"
do

    time python -u generate_ref.py \
        --device 0  --seed 0 \
        --datasplit ${datasplit} \
        --data_dir ${data_dir} \
        --model_name ${model_name} \
        --use_large_feats

done

echo "Job completed at $(date)"
