#!/bin/bash
#SBATCH --job-name              vlt5-mmt-train
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   30G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/vlt5-mmt-train.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/vlt5-mmt-train.%j.err
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

dataset=ambig
src=en
tgt=zh
# The name of experiment
name=VLT5-$dataset-$src-$tgt

# hyperparameters
learning_rate=1e-3
batch_size=30
gradient_accumulation=2

output=snap/Multi30K/$name/lr-${learning_rate}_bs-$((batch_size * gradient_accumulation))

if [ "$dataset" = "m30k" ]; then
    epochs=40
    test=test_2016_flickr,test_2017_flickr
else
    epochs=100
    test=test
fi

# w/ pre-training
PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47771 \
    src/mmt.py \
        --distributed --multiGPU \
        --dataset $dataset \
        --target $tgt \
        --train train \
        --valid val \
        --test $test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr $learning_rate \
        --epochs $epochs \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation \
        --max_text_length 40 \
        --gen_max_length 40 \
        --do_lower_case \
        --comment ${dataset}_lr-${learning_rate}_bs-$((batch_size * gradient_accumulation))

# w/o pre-training
# PYTHONPATH=$PYTHONPATH:./src \
# python -m torch.distributed.launch \
#     --nproc_per_node=$1 \
#     --master_port 47770 \
#     src/mmt.py \
#         --distributed --multiGPU \
#         --dataset $dataset \
#         --target $tgt \
#         --train train \
#         --valid val \
#         --test $test \
#         --optim adamw \
#         --warmup_ratio 0.1 \
#         --clip_grad_norm 5 \
#         --lr $learning_rate \
#         --epochs 40 \
#         --num_workers 4 \
#         --backbone 't5-base' \
#         --output $output ${@:2} \
#         --num_beams 5 \
#         --batch_size $batch_size \
#         --gradient_accumulation_steps $gradient_accumulation \
#         --max_text_length 40 \
#         --gen_max_length 40 \
#         --do_lower_case \
#         --comment ${dataset}_lr-${learning_rate}_bs-$((batch_size * gradient_accumulation))