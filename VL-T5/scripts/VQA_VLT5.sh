#!/bin/bash
#SBATCH --job-name              vlt5-vqa-train
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   50G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/vlt5-vqa-train.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/vlt5-vqa-train.%j.err
#SBATCH --mail-type		NONE
#SBATCH --mail-user		yc27434@connect.um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh

#Adding modules
# module add cuda/9.2.148
# module add amber/18/gcc/4.8.5/cuda/9

ulimit -s unlimited

#Your program starts here 
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# The name of experiment
name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa.py \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 20 \
        --valid_batch_size 100 \