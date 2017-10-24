import numpy as np
import tensorflow as tf 

# Read the test data
test_x = np.genfromtxt('test_data.csv', dtype=np.uint8, delimiter=',')
test_y = np.genfromtxt('test_hand.csv', dtype=np.uint8, delimiter=',')

test_dat = np.zeros((len(test_x), 52))

for i in range(len(test_x)):
    for j in range(0,10,2):
        test_dat[i, (test_x[i,j]-1)*13 + (test_x[i,j+1]-1)]=1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 52   # 4 * 13 cards
num_classes = 10 # number of classes 

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[offset + labels_dense.ravel()] = 1
    return labels_one_hot

def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Learning rate 
lr = tf.placeholder(tf.float32)

# Construct model
logits = neural_net(X)

# Evaluate model (with test logits, for dropout to be disabled)
pred_classes = tf.argmax(logits, axis=1)

# Create saver
saver = tf.train.Saver()

with tf.Session() as sess:

    # Run the initializer
    saver.restore(sess, "model/98/card_model-6000")

    # Calculate accuracy for the test set
    pred = sess.run(pred_classes, feed_dict={X: test_dat, Y: dense_to_one_hot(test_y)})
    test_acc = np.sum(1*(pred == test_y))*1./len(test_y)

    print "Testing Accuracy:", test_acc

    file = open('output.txt', 'w')
    for i in pred:
    	file.write(str(i) + "\n")