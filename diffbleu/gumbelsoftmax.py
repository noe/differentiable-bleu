""" Batched Gumbel SoftMax.

Taken from https://github.com/tokestermw/text-gan-tensorflow/blob/master/distributions.py

Paper:
    CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX
    https://arxiv.org/abs/1611.01144

    The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    https://arxiv.org/abs/1611.00712

Code:
    https://github.com/ericjang/gumbel-softmax
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from collections import namedtuple

try:
    from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import ExpRelaxedOneHotCategorical
except:
    from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import _ExpRelaxedOneHotCategorical
    print("TensorFlow native concrete distribution (this version doesn't work).")


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        vocab_size = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.one_hot(tf.argmax(y, -1), vocab_size), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

