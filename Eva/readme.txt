mnist-challenge-master:(modified from https://github.com/MadryLab/mnist_challenge)
    1. config parameters and paths in config.json, mainly model_path and eps
    2. run train.py to train tensorflow model for mnist dataset(with adversial examples), the models are saved in checkpoints in ./models folder
    2b. model with eps=0 and 0.3 can also be downloaded using fetch_model.py, in this way: python fetch_model.py natural(or adv_trained)
    3a. when analyzing characteristic of model, run gbt_eval.py to evaluate with trained model through the train/test set(output paths can be set in the py file, while input and other config in config.json)
    3b. when runing for clever score, run model_converter.py to get the .h5 file containing structure and weights of the model. Paths are set at the head of the py file. 
        by default .5h model files are also stored in ./models folder

robustness-master:
    1. install robustness package using pip or other similar tools
    2. download model from github page https://github.com/MadryLab/robustness, 
        or train model with robustness, e.g. 
            python -m robustness.main --dataset restrict_imagenet  --adv-train 0 --arch resnet50 --out-dir logs/checkpoints/dir/
        for a naturally trained resnet-50 model for cifar dataset, 
        more parameters see 
            https://robustness.readthedocs.io/en/latest/example_usage/cli_usage.html#training-a-standard-nonrobust-model
    3. evaluate through the model trained using run.py. 
        by default .pt model files are put together with run.py, and output files are stored in ./train_eval_results foler

graspel:
    (use matlab)
    1. run runGraspel.m to get the knn graph of the predictions of the models. 
        path of target files(prediction results and labels, and output matrix) can be set in runGrapel.m. 
    2. run robustnessCIFAR.m/robustnessMNIST.m to read a matrix, calculate riemannian distances and sort edges and nodes with the distances. 
    3(optional). after running a complete set of matrices, gbt_getTopNodeList.m can save the top N node/edge into csv file. 

CLEVER:(modified from https://github.com/huanzhang12/CLEVER)
    1. run `python3 collect_gradients.py --data mnist --model_name 2-layer --target_type 16 --numimg 10` to get gredients, you can modify model_name to "2-layer" (MLP), "normal" (7-layer CNN), "distilled" (7-layer CNN with defensive distillation)
    1b(optional). to calculate SPADE-Guided gredients, add `--ids ids_cifar_cnn.csv` to 1. you can switch all csv file to ids to calculate different network gredients. topnode stands for k=10.
    2. run `python3 clever.py --untargeted ./lipschitz_mat/mnist_2-layer/` to get clever score. different networks have different address.

CLEVER_ourmodel:(modified from https://github.com/huanzhang12/CLEVER)
    1. same as CLEVER, but model_name options are mnist_0, mnist_01, mnist_02, mnist_03
