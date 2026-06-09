#!/bin/bash

module purge
module load modules/2.2
module load python 
module load cuda cudnn nccl

source /mnt/home/alu10/envs/jupyter-gpu/bin/activate

# Slurm command: sbatch -p gpu --gpus=4 -C a100-40gb --cpus-per-gpu=4 run_training.sh

python -u `which torchrun` \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 4 \
    train.py --yaml_conf /mnt/ceph/users/ajacinto/nuclear_segmentation/confs/train_convnext_unet_large.yaml

