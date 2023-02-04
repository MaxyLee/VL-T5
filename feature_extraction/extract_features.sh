#!/bin/bash
#SBATCH --job-name              detectron2-feature-extraction
#SBATCH --partition             gpu-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   16G
#SBATCH --gres                  gpu:1
#SBATCH --output                /data/home/yc27434/projects/mmt/logs/detectron2-feature-extraction.%j.out
#SBATCH --error                 /data/home/yc27434/projects/mmt/logs/detectron2-feature-extraction.%j.err
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

# DATAROOT=/data/home/yc27434/projects/mmt/data/MuCoW
# DATASET_NAME=mucow-mmt

DATAROOT=/data/home/yc27434/projects/mmt/data/multisense
DATASET_NAME=multisense

python mmt_proposal.py --dataroot $DATAROOT --dataset_name $DATASET_NAME