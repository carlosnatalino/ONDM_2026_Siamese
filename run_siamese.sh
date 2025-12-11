#!/bin/sh
#SBATCH -A NAISS2025-5-426 -p alvis --time=01:00:00
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
FULL_PATH=/mimer/NOBACKUP/groups/naiss2025-5-426/icmlcn/siamese/$FOLDER_NAME
mkdir -p $FULL_PATH

python siamese.py \
    --embedding_dim 512 \
    --batch_size 128 \
    --augment True \
    --network cnn \
    --early_stopping_patience 25 \
    --test_size 0.2 \
    --val_size 0.2 \
    --epochs 10 \
    --save_dir $FULL_PATH \
    --data-dir /mimer/NOBACKUP/groups/naiss2025-5-426/datasets/DAS-dataset/data

cp $HOME/slurm-${SLURM_JOB_ID}.out $FULL_PATH/slurm-${SLURM_JOB_ID}.out
