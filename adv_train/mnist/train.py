from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import sys
import math
import pickle
import shutil
import argparse
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from numpy.random import permutation
import random
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

parser = argparse.ArgumentParser(description='adv_training')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--topk', type=int, default=5000)
parser.add_argument('--small_eps', type=float, default=0.2)
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--method', type=str, default="spade",
        help="choose training method, [spade, random, pgd]")


args = parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.device)  # specify which GPU(s) to be used

with open('config.json') as config_file:
    config = json.load(config_file)

config["epsilon"] = args.eps

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
x_train = mnist.train.images
y_train = mnist.train.labels
num_samples = x_train.shape[0]
batch_size = config['training_batch_size']
num_batches = int(math.ceil(num_samples/batch_size))
node_epsilon = args.eps * np.ones(x_train.shape[0])
if args.method.startswith("pgd"):
    config["model_dir"] = "models/{}_{}".format(args.method, args.eps)
elif args.method == "spade":
    config["model_dir"] = "models/pgd-{}_{}_{}".format(args.method, args.small_eps, args.eps)
    with open("node_spade_score.pkl", 'rb') as fin :
        node_score = (pickle.load(fin)).reshape(-1,)
        idx = (-node_score).argsort()
        sub_idx = idx[args.topk:]
        node_epsilon[sub_idx] = args.small_eps
elif args.method.startswith("random"):
    config["model_dir"] = "models/pgd-{}_{}_{}".format(args.method, args.small_eps, args.eps)
    node_epsilon = np.asarray(random.choices([args.small_eps, args.eps], k=num_samples))

np.random.seed()
x_train = np.concatenate((x_train, node_epsilon.reshape(-1,1)), axis=1)
print("images with epsilon: ", x_train.shape)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for e in range(args.num_epoch):
        perm = permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]
        for b in range(num_batches):
            bstart = b*batch_size
            bend = min(bstart+batch_size, num_samples)

            x_eps_batch = x_train[bstart:bend,:]
            eps_batch = x_eps_batch[:,-1].reshape(-1,1)
            x_batch = x_eps_batch[:,:-1]
            y_batch = y_train[bstart:bend]

            # Compute Adversarial Perturbations
            start = timer()
            x_batch_adv = attack.perturb(x_batch, y_batch, sess, eps_batch)
            end = timer()
            training_time += end - start

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}

            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            # Output to stdout
            if b % num_output_steps == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
                print('Epoch {}, Step {}:    ({})'.format(e, b, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
                if b != 0:
                    print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                    training_time = 0.0
            # Tensorboard summaries
            if b % num_summary_steps == 0:
                summary = sess.run(merged_summaries, feed_dict=adv_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))

            # Write a checkpoint
            if b % num_checkpoint_steps == 0:
                saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

            # Actual training step
            start = timer()
            sess.run(train_step, feed_dict=adv_dict)
            end = timer()
            training_time += end - start
