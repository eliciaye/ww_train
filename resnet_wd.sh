#!/bin/bash
##SBATCH --array=80-96
#SBATCH --array=1-2,43-44
##SBATCH --array=51-62
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/ww_train
#SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,manchester,steropes,bombe,pavia,luigi
##SBATCH --nodelist=freddie
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet_wd1e-4alpharatio_lrconst0.005_%A_%a.out

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
init_lr=0.005
# init_lr=$(echo $cfg | cut -f 3 -d ' ')
init_wd=1e-4

seed=23

SAVE=/data/eliciaye/val_experiments/ww/cifar100/resnet$depth-$width_frac-lr$init_lr-wd$init_wd

echo $SAVE
mkdir -p $SAVE

datadir=/work/eliciaye
epochs=200
python resnet_train.py --wd_alpha_schedule --constant_lr --depth $depth --checkpoint $SAVE --width_frac $width_frac --wd $init_wd --lr $init_lr --epochs $epochs --seed $seed
echo "All done."
