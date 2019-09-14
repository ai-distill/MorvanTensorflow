import tensorflow as tf
state = tf.Variable(0, name='counter')
print(state.name)
one = tf.constant(1)
new_value = tf.add(state, one)
# 把new_value分配给state
update = tf.assign(state, new_value)

# tesorflow中，如果设定了变量，接下来是最重要的一步。一定要初始化所有变量才会激活
init = tf.initialize_all_variables()

with tf.Session() as sess:
    # 定义好变量以及初始化sess之后，一定要记得sess初始化
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))