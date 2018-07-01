import os

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf


BATCH_SIZE = 128
VOCAB_SIZE = 50000
EMBED_SIZE = 128
NUM_SAMPLED = 64
LEARNING_RATE = 0.01

def word2vec(batch_gen):
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.placeholder(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')

    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / (EMBED_SIZE ** 0.5)), name='nce_weight')

        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed,
                                             num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE), name='loss')

        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

