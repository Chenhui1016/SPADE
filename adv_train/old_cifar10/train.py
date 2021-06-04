from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
import pickle
import argparse
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import random

from model import Model
import cifar10_input
from pgd_attack import LinfPGDAttack

parser = argparse.ArgumentParser(description='adv_training')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--topk', type=int, default=20000)
parser.add_argument('--eps', type=float, default=8.0)
parser.add_argument('--small_eps', type=float, default=6.0)
parser.add_argument('--method', type=str, default="spade",
        help="choose training method, [spade, random, pgd]")


args = parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.device)  # specify which GPU(s) to be used


with open('config.json') as config_file:
    config = json.load(config_file)
config["epsilon"] = args.eps

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

node_epsilon = args.eps * np.ones(50000)
if args.method.startswith("pgd"):
    config["model_dir"] = "models/{}_{}".format(args.method, args.eps)
elif args.method=="spade":
    config["model_dir"] = "models/pgd-{}_{}_{}".format(args.method, args.small_eps, args.eps)
    with open("node_spade_score.pkl", 'rb') as fin :
        node_score = (pickle.load(fin)).reshape(-1,)
        idx = (-node_score).argsort()
    sub_idx = idx[args.topk:]
    node_epsilon[sub_idx] = args.small_eps
elif args.method=="random":
    config["model_dir"] = "models/pgd-{}_{}_{}".format(args.method, args.small_eps, args.eps)
    node_epsilon = np.asarray(random.choices([args.small_eps, args.eps], k=50000))


# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path, node_epsilon.reshape(-1,1,1,1), config["epsilon"])
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['num_steps'],
                       config['step_size'],
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
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch, eps_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

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
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start
