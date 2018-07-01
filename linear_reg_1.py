import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd

DATA_FILE = './data/fire_theft.xls'


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w1 = tf.Variable(0.0, name='w1')
#w2 = tf.Variable(0.0, name='w2')
#w3 = tf.Variable(0.0, name='w3')
b = tf.Variable(0.0, name='bias')

#Y_predicted = X * X * X * w3 + X * X * w2 + X * w1 + b
Y_predicted = X * w1 + b

# loss = tf.square(Y - Y_predicted, name='loss')
loss = huber_loss(Y, Y_predicted)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./my_graph/linear_reg', sess.graph)

    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l1 = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l1

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()

    #w3_value, w2_value, w1_value, b_value = sess.run([w3, w2, w1, b])
    w1_value, b_value = sess.run([w1, b])

X = data.T[0]
Y = data.T[1]
plt.plot(X, Y, 'bo', label='Real Data')
plt.plot(X, X * w1_value + b_value, 'r', label='Predicted Data')
plt.legend()
plt.show()





