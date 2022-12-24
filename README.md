Use `resnet_train.py`

CIFAR100 data loaders in `data.py`
ResNet models in `resnet_widths_all.py`, selectable depth and customizable width
`train()` and `test()` functions in `training.py`
Optimizer customizations in `utils.py`

Customize Arguments (for default values, see the ArgParser section in `resnet_train.py`)
`--lr`: initial learning rate (follow SGD)
`--wd`: initial weight decay, constant for now
`--width_frac`: percentage of original model width, can be `>1`
`--sample_evals`: sample eigenvalues and replace during ww SVD analysis
`--lr_rewind`: learning rate rewinding
`--depth`: select depth of ResNet
`--temp_balance_lr`: layerwise learning rate assignment
`--temp_balance_wd`: layerwise weight decay assignment
