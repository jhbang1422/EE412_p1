import numpy as np
import tensorflow as tf 

flags = tf.flags
flags.DEFINE_string("mode", "test", "Test or train")
flags.DEFINE_string("weights", "model/98/card_model-6000", "filename of the weights for test mode")

# Parameters
num_steps = 6000
batch_size = 256
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

    # shuffle the data if we don't have enough left, then go back to start
    if ctr*batch_size > len(data)-batch_size: 
        data = np.random.permutation(data)
        ctr = -1
    ctr += 1
    batch_data = data[ctr*batch_size:(1 + ctr)*batch_size, :(data.shape[1]-1)]
    batch_labels = dense_to_one_hot(data[ctr*batch_size:(1 + ctr)*batch_size, data.shape[1]-1]) 
    
    return batch_data, batch_labels

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[offset + labels_dense.ravel()] = 1
    return labels_one_hot

class RankClassifier:
    def __init__(self, mode):
        self.learning_rate = 0.01

        # tf Graph input
        self.X = tf.placeholder("float", [None, num_input])
        self.Y = tf.placeholder("float", [None, num_classes])

        # Learning rate 
        self.lr = tf.placeholder(tf.float32)

        # Construct model
        self.logits = self.neural_net(self.X)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        self.pred_classes = tf.argmax(self.logits, axis=1)
        self.correct_pred = tf.equal(self.pred_classes, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Create saver
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        if mode == "test":
            self.saver.restore(self.sess, flags.FLAGS.weights)
        

    def neural_net(self, x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.layers.dense(x, n_hidden_1)
        layer_1 = tf.nn.relu(layer_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.layers.dense(layer_1, n_hidden_2)
        layer_2 = tf.nn.relu(layer_2)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.layers.dense(layer_2, num_classes)
        return out_layer

    def classify(self, test_data):
        pred = self.sess.run(self.pred_classes, feed_dict={self.X: test_data})
        return pred

    def train(self):

        for step in range(1, num_steps+1):
            batch_x, batch_y = get_batch(batch_size)

            # Run optimization
            self.sess.run(self.train_op, feed_dict={self.X: batch_x, 
                                                    self.Y: batch_y, 
                                                    self.lr: self.learning_rate})

            # decrease learning rate
            if step % 5000 == 0:
                self.learning_rate *= 0.5

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = self.sess.run([self.loss_op, self.accuracy], 
                    feed_dict={self.X: batch_x, self.Y: batch_y})

                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        # Save the model
        self.saver.save(self.sess, 'model/card_model', global_step=step)
        
        print "Optimization Finished!"

# Read the training data
dset = np.genfromtxt('train_data.csv', dtype=np.uint8, delimiter=',')

# Read the test data
test_x = np.genfromtxt('test_data.csv', dtype=np.uint8, delimiter=',')
test_y = np.genfromtxt('test_hand.csv', dtype=np.uint8, delimiter=',')

data = np.zeros((len(dset), num_input + 1), dtype=np.int8)
test_dat = np.zeros((len(test_x), num_input))

# Extract features from the training data
# 1: card is present, 0: card is absent
# 0-12: suit 2, 13-25: suit 2, 26-38: suit 3, 39-51: suit 4
for i in range(len(dset)):
    for j in range(0,10,2):
        data[i, (dset[i,j]-1)*13 + (dset[i,j+1]-1)]=1
    data[i, 52] = dset[i, 10]

for i in range(len(test_x)):
    for j in range(0,10,2):
        test_dat[i, (test_x[i,j]-1)*13 + (test_x[i,j+1]-1)]=1

rc = RankClassifier(mode=flags.FLAGS.mode)

if flags.FLAGS.mode == "train":
    rc.train()

pred = rc.classify(test_dat)
test_acc = np.sum(1*(pred == test_y))*1./len(test_y)

print "\nTesting Accuracy:", test_acc

file = open('output.txt', 'w')
for i in pred:
    file.write(str(i) + "\n")

print "Output saved in output.txt"