#!/bin/bash
#SBATCH --job-name              t5-mmt-train
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   30G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/t5-mmt-train.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/t5-mmt-train.%j.err
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

src=en
tgt=zh
# The name of experiment
name=T5-$src-$tgt

output=snap/Multi30K/$name

# train
PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/mmt.py \
        --distributed --multiGPU \
        --debug \
        --use_vision False \
        --target $tgt \
        --train train \
        --valid val \
        --test test_2016_flickr,test_2017_flickr \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 40 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 30 \
        --max_text_length 40 \
        --gen_max_length 40 \
        --do_lower_case \

# debug
# python src/mmt.py \
#         --debug \
#         --target $tgt \
#         --train train \
#         --valid val \
#         --test test_2016_flickr,test_2017_flickr \
#         --optim adamw \
#         --warmup_ratio 0.1 \
#         --clip_grad_norm 5 \
#         --lr 5e-5 \
#         --epochs 1 \
#         --num_workers 1 \
#         --backbone 't5-base' \
#         --output $output ${@:2} \
#         --load snap/pretrain/VLT5/Epoch30 \
#         --num_beams 5 \
#         --batch_size 1 \
#         --max_text_length 40 \
#         --gen_max_length 40 \
#         --do_lower_case \