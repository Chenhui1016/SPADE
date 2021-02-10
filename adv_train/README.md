SPADE-adv\_train
===============================

The adversarial training kernel of SPADE.


Requirements
------------
* tensorflow <= 1.15
* googledrivedownloader (Only required if you fetch the pre-trained models)


Usage
-----

**MNIST Usage**

1. `cd mnist/`

2. `python fetch_model.py --method spade` (Only required if you fetch the pre-trained model)

3. `python train.py --device 0 --method spade` (Only required if you train the model from scratch)

4. `python eval.py --device 0 --method spade`


**CIFAR10 Usage**

1. `cd cifar10/`

2. `python fetch_model.py --method spade` (Only required if you fetch the pre-trained model)

3. `python train.py --device 0 --method spade` (Only required if you train the model from scratch)

4. `python eval.py --device 0 --method spade`
