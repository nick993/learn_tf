import tensorflow as tf
import numpy as np

# Create Variable with Scalar Value
a = tf.Variable(2, name='Scalar')
a1 = tf.Variable([2, 3], name='Vector')
a2 = tf.Variable([[2, 3], [4, 5]], name='Matrix')
w = tf.Variable(tf.zeros([784, 10]), name='W')

assign_a_op = a.assign(100)

# Have to initialize the variables
init = tf.global_variables_initializer()

# To initialize variables subset
init_sb = tf.variables_initializer([a, a1], name='init_a_a1')

# Original Value is 2
b2 = tf.Variable(2, name='initB')

# Assign an op which multiplies by 2
op_mul_2 = b2.assign(tf.multiply(b2, 2))

# Each session maintains its own copy of variable
# To control graph flow tf.Graph.control_dependencies([a, b, c]) : other ops wiill only run after a, b, c have executed.

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphVariables', sess.graph)

    # To initialize the variables
    sess.run(init)
    sess.run(init_sb)

    # To initialize a single variable
    sess.run(w.initializer)

    # initial value of a so not displaying 100
    print('A :', a.eval())

    sess.run(assign_a_op)
    print('Assigned A :', a.eval())

    print('Variable a2', a2.eval())

    sess.run(b2.initializer)
    print('Printing 2 Multiples', b2.eval())
    sess.run(op_mul_2)
    print('Printing 2 Multiples', b2.eval())
    sess.run(op_mul_2)
    print('Printing 2 Multiples', b2.eval())

writer.close()

