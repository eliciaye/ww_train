#!/bin/bash
datadir=/work/eliciaye
epochs=200
depth=50
seed=7777

# (fit, metric) = {(E_TPL, alpha), (E_TPL, Lambda), (E_TPL, D), (PL, D)}
fit=PL
metric=alpha

# temp_balance_lr = {tbr, avg, sqrt}
temp_balance_lr=tbr

lrs=(0.025 0.05 0.075 0.1 0.15 0.2)
widths=(0.25 0.5 1 1.5 2 2.5)

for i in "${!lrs[@]}"
do
    for j in "${!widths[@]}"
    do 
        lr=${lrs[$i]}
        width_frac=${widths[$j]}
        NAME=$fit-$metric-$temp_balance_lr-resnet$depth-$width_frac-$lr
        SAVE=/data/eliciaye/val_experiments/cifar100/$NAME
        mkdir -p $SAVE
        idx=$((6*$i+$j))
        CUDA_VISIBLE_DEVICES=$j python resnet_train.py --fit $fit --metric $metric --temp_balance_lr $temp_balance_lr --depth $depth --width_frac $width_frac --checkpoint $SAVE --lr $lr --epochs $epochs --seed $seed >>$fit-$metric-$temp_balance_lr-$idx.out>&1 &
    done
      
done

echo "All done."