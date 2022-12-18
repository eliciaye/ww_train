#!/bin/bash
#SBATCH --array=73-78,1,6,11,16,21,2,7,12,17,22
##SBATCH --array=51-62
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/ww_train
#SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,manchester,bombe,pavia,como,luigi
##SBATCH --nodelist=freddie
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet_wdlr_avg_%A_%a.out

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
cd /home/eecs/eliciaye/ww_train

function=avg

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p resnet_config.txt)
idx=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
Net=$(echo $cfg | cut -f 3 -d ' ')
init_lr=$(echo $cfg | cut -f 4 -d ' ')

SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-$function-init$init_lr-wd
seed=7

echo $SAVE
datadir=/work/eliciaye
epochs=200
mkdir -p $SAVE
# python resnet_weight_decay.py --checkpoint $SAVE --lr $init_lr --Net $Net --epochs $epochs --function $function --seed $seed
# python resnet_weight_decay.py --sample_wd --checkpoint $SAVE --lr $init_lr --Net $Net --epochs $epochs --function $function --seed $seed
python resnet_weight_decay.py --lr_schedule --checkpoint $SAVE --lr $init_lr --Net $Net --epochs $epochs --function $function --seed $seed
echo "All done."
