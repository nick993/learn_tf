import numpy as np
import tensorflow as tf

sess = tf.Session()
a = tf.add(3, 4)
b = tf.subtract(3, 4)
c = tf.multiply(a, b)
print('Add 3 and 4', sess.run(a))
print('Subtract 3 and 4', sess.run(b))
print('3 squared - 4 squared', sess.run(c))

sess.close()

g = tf.Graph()
with g.as_default():
    ar1 = tf.constant([1, 2, 3, 4], name='a')

    ar2 = tf.constant([1, 2, 3, 4], name='b')
    # Dimension should be same
    ar3 = tf.multiply(ar1, ar2)

    a = tf.constant(2, shape=[2, 2], name='a')
    b = tf.constant([2,4], shape=[3, 3], name='b')
    ar4 = tf.constant([2, 1], name='ar4')
    br4 = tf.constant([[0, 1], [2, 3]], name='br4')
    res4 = tf.add(ar4, br4, name='res4')

    # Zero MAtrix
    z1 = tf.zeros([2, 3], tf.int32)

    # Zeros like a tensor
    z2 = tf.zeros_like(br4)

    # Similarly for ones
    o1 = tf.ones([1, 4], tf.float64)
    o2 = tf.ones_like(br4)

    # fill
    f1 = tf.fill([4, 6], 3)

    # Throws Error
    # a = tf.constant(2, shape=[2, 2], verify_shape=True)

    # constants as sequence (start, stop, num)
    seq1 = tf.linspace(10.0, 14.0, 4)

    # constants as range (start, limit, delta)
    seq2 = tf.range(10.0, 14.0, 0.3)
    seq3 = tf.range(6)

    # Random Numbers and mean partial array selection
    r1 = tf.random_normal([500, 500], mean=0.0, stddev=1.0, name='RandomNumbers')
    mmr_1 = tf.random_normal([1000, 2], mean=[0.0, 1.0], stddev=[1.0, 2.0], name='MultipleRandomNumbers')
    # Truncated to 2 times stddev
    r1_truncated = tf.truncated_normal([500, 500], mean=0.0, stddev=1.0, name='TruncatedNormal')
    r_uniform = tf.random_uniform([500, 500], minval=0.0, maxval=3.0, seed=1132)
    m1 = tf.reduce_mean(r1, name='FullMean')
    mt1 = tf.reduce_mean(r1_truncated, name='MeanTruncatedNormal')
    mu1 = tf.reduce_mean(r_uniform, name='MeanUniformNormal')
    m1_1 = tf.reduce_mean(r1[1, :], name='FirstVectorMean')

    # Shuffle
    m1 = tf.linspace(start=1.0, stop=4.0, num=10, name='seq1')
    m2 = tf.linspace(start=4.0, stop=8.0, num=10, name='seq2')
    m3 = tf.linspace(start=8.0, stop=12.0, num=10, name='seq3')
    m_addn = tf.add_n([m1, m2, m2], name='AddNVectors')
    m = tf.concat([m1, m2, m3], 0, name='ConcatSequence')
    m_shuffle = tf.random_shuffle(m, name='RandomShuffle')

    # Random Crop
    c1 = tf.range(100, name='100Range')
    c1_crop = tf.random_crop(c1, [10], name='10RangeCrop')

    # Multinomial
    mm1 = tf.constant(np.random.normal(size=(3, 4)))
    mm_trans1 = tf.multinomial(mm1, 5)

    # Random Seed
    seed1 = tf.set_random_seed(1000)

    # Tensorflow Data Type : 0-d tensor i.e Scalar, 1-d Vector, 2-d i.e matrix

    # Random Gamma
    # tf.random_gamma

with tf.Session(graph=g) as sess:
    writer = tf.summary.FileWriter("./graphsLTF1", sess.graph)
    print('ar3', sess.run(ar3))
    print('res4', sess.run(res4))
    print('z2', sess.run(z2))
    print('o2', sess.run(o2))
    print('f1', sess.run(f1))
    print('seq1', sess.run(seq1))
    print('seq2', sess.run(seq2))
    print('seq2', sess.run(seq2))
    print('seq3', sess.run(seq3))
    print('Mean full :', sess.run(m1), ' : Mean first vector: ', sess.run(m1_1))
    print('Multiple Dist Mean 0 and 1', sess.run(mmr_1))
    print('Mean full Truncated Normal :', sess.run(mt1))
    print('Mean Uniform :', sess.run(mu1))
    print('Sequence :', sess.run(m))
    print('Random Shuffle :', sess.run(m_shuffle))
    print('10 Range Crop :', sess.run(c1_crop))
    print('Multinomial', sess.run(mm_trans1))
    # Constants are stored in graph
    # print(sess.graph.as_graph_def())

writer.close()






