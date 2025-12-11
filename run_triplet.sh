#!/bin/sh
#SBATCH -A NAISS2025-5-426 -p alvis --time=24:00:00
#SBATCH --gpus-per-node=V100:1

module load PyTorch-bundle
module load matplotlib/3.7.2
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a

cp -av $HOME/icmlcn/ $TMPDIR

cd $TMPDIR/icmlcn

# TODO:
# 1. increase epochs
# 2. increase batch size

FOLDER_NAME=$(date +"%Y%m%d_%H%M%S")
mkdir -p /mimer/NOBACKUP/groups/naiss2025-5-426/icmlcn/triplet/$FOLDER_NAME

python siamese_triplet.py \
    --batch_size 256 \
    --train_classes 0 2 4 6 8 \
    --early_stopping_patience 25 \
    --epochs 20 \
    --triplet_pretrain_epochs 50 \
    --save_dir /mimer/NOBACKUP/groups/naiss2025-5-426/icmlcn/triplet/$FOLDER_NAME \
    --data-dir /mimer/NOBACKUP/groups/naiss2025-5-426/datasets/DAS-dataset/data

cp $HOME/slurm-${SLURM_JOB_ID}.out /mimer/NOBACKUP/groups/naiss2025-5-426/icmlcn/triplet/$FOLDER_NAME/slurm-${SLURM_JOB_ID}.out
