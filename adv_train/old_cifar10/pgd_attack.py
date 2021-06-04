from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10_input

class LinfPGDAttack:
  def __init__(self, model, num_steps, step_size, random_start, loss_func):
    self.model = model
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess, epsilon):
    if self.rand:
      x = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)

    for i in range(self.num_steps):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - epsilon, x_nat + epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x
