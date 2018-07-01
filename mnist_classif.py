import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

learnning_rate = 0.01
batch_size = 128
n_epochs = 30

mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

X = tf.placeholder(tf.float32, [batch_size, 784], name='X_Placeholder')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y_Placeholder')

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1,10]) , name='bias')

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='loss')
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdagradOptimizer(learnning_rate).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./my_graph/mnist_classif', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    n_batches = int(mnist.train.num_examples/batch_size)

    for i in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch

        print('Average Loss epoch {0} : {1}'.format(i, total_loss/n_batches))

    print('Total Time : {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished')

    # Test the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_pred = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        _, loss_batch, logit_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logit_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_pred += sess.run(accuracy)

    print('Accuracy {0}'.format(total_correct_pred/mnist.test.num_examples))

    writer.close()

# Tensorboard
# tensorboard --logdir=path/to/log-directory