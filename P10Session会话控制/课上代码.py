import tensorflow as tf
import numpy as np
print(np.__version__)
print(tf.__version__)

matrix1 = tf.constant([[3, 3]])     # 定义1行2列的矩阵
matrix2 = tf.constant([[2],         # 定义2行1列的矩阵
                       [2]])
product = tf.matmul(matrix1, matrix2)


# 方法一
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# 方法二：自动关闭session
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

