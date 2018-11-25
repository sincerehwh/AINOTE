
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
img = mnist.train.images[0].reshape(28,28)

# 设置placeholder
X = tf.placeholder(tf.float32,[None,784])
X_img = tf.reshape(X, [-1,28,28,1])
Y = tf.placeholder(tf.float32,[None,10])

kernel1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
L1 = tf.nn.conv2d(X_img,kernel1,strides=[1,1,1,1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")

kernel2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2 = tf.nn.conv2d(L1,kernel2,strides=[1,1,1,1],padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
L2 = tf.reshape(L2,[-1,7 * 7 * 64])
L2 = tf.reshape(L2,[-1,7 * 7 * 64])

W3 = tf.Variable(tf.random_normal([7 * 7 * 64,10],stddev=0.01))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L2, W3) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_pochs = 1
batch_size = 32
print("------------------Start Learning-----------------")

for poch in range(training_pochs):
	avg_cost = 0
	total_batch = int(mnist.train.num_examples / batch_size)
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		feed_dict = {X: batch_xs, Y: batch_ys}
		c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
		avg_cost += c / total_batch
		content = "Epoch@{} || cost={:.9f} ".format(poch+1,avg_cost)
		print(content)

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:',sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))









