import tensorflow as tf
import numpy as np

# 使用placeholder为了外界传值方便

if __name__ == '__main__':
    # 规定2行2列的结构
    # input1 = tf.placeholder(tf.float32, [2, 2])
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))