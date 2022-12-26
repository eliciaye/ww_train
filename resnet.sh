#!/bin/bash
##SBATCH --array=1,6,11,16,21,2,7,12,17,22,26,31,36,41,46,27,32,37,42,47
#SBATCH --array=54,64-72,74-78,81-84,9-12,15-18,21-24,25,27,33-36,39-42
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/ww_train
#SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,manchester,pavia,como,luigi,steropes,bombe
##SBATCH --nodelist=pavia
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet_tbr_new_%A_%a.out

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
cd /home/eecs/eliciaye/ww_train

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p config_lr.txt)
depth=$(echo $cfg | cut -f 1 -d ' ')
width_frac=$(echo $cfg | cut -f 2 -d ' ')
lr=$(echo $cfg | cut -f 3 -d ' ')

SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$width_frac-$lr

echo $SAVE
datadir=/work/eliciaye
epochs=200
mkdir -p $SAVE
arch_max=0.99
seed=5

python resnet_train.py --temp_balance_lr tbr --depth $depth --width_frac $width_frac --checkpoint $SAVE --lr $lr --epochs $epochs --seed $seed
echo "All done."
