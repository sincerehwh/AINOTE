

import tensorflow as tf

# 0.创建模拟数据
x_train = [1,2,3]
y_train = [3,4,5]
l_times = 200000 + 1
l_rate = 0.001
print_step = 200 

# 1.构建目标函数： y = wx + b
w = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))
y_predict = x_train * w + b

# 2.构建损失函数
cost = tf.reduce_mean(tf.square(y_predict - y_train)) 

# 3.采用梯度下降更新权重
optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)

# 优化器
train = optimizer.minimize(cost)

# 4.运行计算图训练
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(l_times):
		sess.run(train)
		if i % print_step == 0:
			content = "step:{} | cost:{} |  w:{}  | b: {}".format(i,sess.run(cost),sess.run(w),sess.run(b))
			print(content)





