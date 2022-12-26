### Use `resnet_train.py`

Optimizer customizations in `utils.py`

CIFAR100 data loaders in `data.py`

`train()` and `test()` functions in `training.py`

ResNet models in `resnet_widths_all.py`, selectable depth and customizable width


### Customize Arguments (for default values, see the ArgParser section in `resnet_train.py`)

`--depth`: select depth of ResNet

`--lr`: initial learning rate (follow SGD)

`--constant_lr`: no learning rate schedule, constant through all training epochs

`--wd`: initial weight decay, constant for now

`--width_frac`: percentage of original model width, can be $>1$

`--sample_evals`: sample eigenvalues and replace during ww SVD analysis

`--lr_rewind`: learning rate rewinding: if layer alpha increases too much from previous epoch, set learning rate schedule back 20 epochs

`--wd_alpha_schedule`: layerwise weight decays by ratio $\frac{\alpha_t}{\alpha_{t-1}}$

`--temp_balance_lr`: layerwise learning rate assignment fn

`--temp_balance_wd`: layerwise weight decay assignment fn
