import numpy as np
import tensorflow as tf

print(np.__version__)
print(tf.__version__)


# 首先创造数据
x_data = np.random.rand(100).astype(np.float32)
print(x_data)
y_data = x_data * 0.1 + 0.3
print(y_data)

# 开始创造tensorflow结构
## Weights可能是一个矩阵，所以开头大写
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
## 偏置开始设置为0
biases = tf.Variable(tf.zeros([1]))
## 预测的y
y = Weights * x_data + biases
## 计算损失函数
loss = tf.reduce_mean(tf.square(y-y_data))
## 神经网络要做的就是建立optimizer减少误差，提高参数的准确度
optimizer = tf.train.GradientDescentOptimizer(0.5)  #0.5指的是学习效率，一般是小于1的数
train = optimizer.minimize(loss)
## 先建立结构，然后初始化，让其活动起来
init = tf.initialize_all_variables()

# 开始会话
sess = tf.Session()
## 此时激活神经网络
sess.run(init)


# 让神经网络一步步开始训练
for step in range(201):
    sess.run(train)
    # 每隔20步打印训练结果
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
