#!/bin/bash
##SBATCH --array=1,6,11,16,21,2,7,12,17,22,26,31,36,41,46,27,32,37,42,47
##SBATCH --array=3-5,8-10,13-15,18-20,23-25,28-30,33-35,38-40,43-45,48-50
#SBATCH --array=79-96
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/ww_train
#SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,manchester,steropes,bombe,pavia,como,luigi
##SBATCH --nodelist=bombe
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet_baseline_%A_%a.out

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
cd /home/eecs/eliciaye/ww_train


cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p resnet_config.txt)
idx=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
Net=$(echo $cfg | cut -f 3 -d ' ')
init_lr=$(echo $cfg | cut -f 4 -d ' ')
seed=9

SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-$init_lr-baseline

echo $SAVE
datadir=/work/eliciaye
mkdir -p $SAVE
arch_max=0.99

cd ../trainer
python resnet_baseline.py --checkpoint $SAVE --lr $init_lr --Net $Net --seed $seed
echo "All done."
