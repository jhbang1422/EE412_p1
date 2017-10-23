import tensorflow as tf 
import numpy as np 


# parameters 
learning_rate = 0.1
num_steps = 500
batch_size= 128
display_step =100

# Network Parameters 
n_hidden_1 = 10 #1st layer number of neurons 
n_hidden_2 = 10 #2nd layer number of neurons 
num_input = 10 
num_classes = 10

#tf input output 
data = np.loadtxt("train_data.csv", delimiter=",", dtype=np.int)
x = data[:,0:-1]
y = data[:,[-1]]

print (x)
# weights & bias 
weights = {
	'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])), 
	'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normnal([n_hidden_2])), 
	'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
	layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	out_layer = tf.matmul(layer_2, weights['out'])+biases['out']
	return out_layer;




with tf.Session() as sess:
	sess.run()
