import numpy as np
import tensorflow as tf

# Read the training data
dset = np.genfromtxt('train_data.csv', dtype=np.uint8, delimiter=',')

# Read the test data
test_x = np.genfromtxt('test_data.csv', dtype=np.uint8, delimiter=',')
test_y = np.genfromtxt('test_hand.csv', dtype=np.uint8, delimiter=',')

data = np.zeros((len(dset), 53), dtype=np.uint8)
test_dat = np.zeros((len(test_x), 52))

# Extract features from the training data
# 1: card is present, 0: card is absent
# 0-12: suit 2, 13-25: suit 2, 26-38: suit 3, 39-52: suit4
for i in range(len(dset)):
    for j in range(0,10,2):
        data[i, (dset[i,j]-1)*13 + (dset[i,j+1]-1)]=1
    data[i, 52] = dset[i, 10]

for i in range(len(test_x)):
    for j in range(0,10,2):
        test_dat[i, (test_x[i,j]-1)*13 + (test_x[i,j+1]-1)]=1

# Parameters
learning_rate = 0.01
num_steps = 6000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 52   # 4 * 13 cards
num_classes = 10 # number of classes 


ctr = -1
def get_batch(batch_size):
    """Get a batch of data."""
    global ctr
    global data

    # shuffle the data if we don't have enough left then go back to start
    if ctr*batch_size > len(data)-batch_size: 
        data = np.random.permutation(data)
        ctr = -1
    ctr += 1
    batch_data = data[ctr*batch_size:(1 + ctr)*batch_size, :(data.shape[1]-1)]
    batch_labels = dense_to_one_hot(data[ctr*batch_size:(1 + ctr)*batch_size, data.shape[1]-1]) 
    
    return batch_data, batch_labels

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
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

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
pred_classes = tf.argmax(logits, axis=1)
correct_pred = tf.equal(pred_classes, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = get_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % 2000 == 0:
            learning_rate *= 0.5

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    saver.save(sess, 'model/card_model', global_step=step)

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_dat,
                                      Y: dense_to_one_hot(test_y)}))
    
    pred = sess.run(pred_classes, feed_dict={X: test_dat, Y:dense_to_one_hot(test_y)})

    file = open('output.txt', 'w')
    for i in pred:
    	file.write(str(i) + "\n")
