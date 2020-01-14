import numpy as np
import tensorflow as tf

inputs = [[1,0,2],[3,2,4]]
inputs = np.array(inputs)
A = tf.sign(inputs)
B = tf.reduce_sum(A, reduction_indices=1)
with tf.Session() as sess:
    print(sess.run(A))
    print(sess.run(B))