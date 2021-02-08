SPADE-adv\_train
===============================

The adversarial training kernel of SPADE.


Requirements
------------
* tensorflow <= 1.15


Usage
-----

**MNIST Usage**

1. `cd mnist/`

2. `python train.py --device 0 --method spade` (Only required if you want to train the model from scratch)

3. `python eval.py --device 0 --method spade`


**CIFAR10 Usage**

1. `cd cifar10/`

2. `python train.py --device 0 --method spade` (Only required if you want to train the model from scratch)

3. `python eval.py --device 0 --method spade`
