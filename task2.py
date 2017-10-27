import numpy as np
import tensorflow as tf 
from sklearn.metrics import confusion_matrix
import time

flags = tf.flags
flags.DEFINE_string("mode", "test", "Test or train")
flags.DEFINE_string("weights", "model/99/card_model-6000", "filename of the weights for test mode")

# Parameters
num_steps = 6000
batch_size = 256
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 58#69   # 4 * 13 cards
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
'''
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[offset + labels_dense.ravel()] = 1
    return labels_one_hot
'''
def dense_to_one_hot(labels_dense, num_clasees=10):
    num_labels = len(labels_dense)
    labels_one_hot = np.zeros((num_labels,num_classes))
    for i in range(num_labels):
        labels_one_hot[i][int(labels_dense[i])]=1
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
# 52-55: number of cards per suit
# 55-68: number of cards per number
since2 = time.time()
for i in range(len(dset)):
    for j in range(0,10,2):
        data[i, (dset[i,j]-1)*13 + (dset[i,j+1]-1)]=1
    # for j in range(0,4):
    #     data[i,52+j] = np.sum((data[i,j*13:(j+1)*13]))
    # for j in range(0,13):
    #     data[i,56+j] = data[i,j] + data[i,13+j] + data[i,26+j] + data[i,39+j]
    data[i, 52] = np.sum(data[i, 0:13]*data[i, 13:26])
    data[i, 53] = np.sum(data[i, 26:39]*data[i, 39:52])
    data[i, 54] = np.sum(data[i, 13:26]*data[i, 26:39])
    data[i, 55] = np.sum(data[i, 0:13]*data[i, 39:52])
    data[i, 56] = np.sum(data[i, 0:13]*data[i, 26:39])
    data[i, 57] = np.sum(data[i, 13:26]*data[i, 39:52])
    data[i, num_input] = dset[i, 10]

for i in range(len(test_x)):
    for j in range(0,10,2):
        test_dat[i, (test_x[i,j]-1)*13 + (test_x[i,j+1]-1)]=1
    test_dat[i, 52] = np.sum(test_dat[i, 0:13]*test_dat[i, 13:26])
    test_dat[i, 53] = np.sum(test_dat[i, 26:39]*test_dat[i, 39:52])
    test_dat[i, 54] = np.sum(test_dat[i, 13:26]*test_dat[i, 26:39])
    test_dat[i, 55] = np.sum(test_dat[i, 0:13]*test_dat[i, 39:52])
    test_dat[i, 56] = np.sum(test_dat[i, 0:13]*test_dat[i, 26:39])
    test_dat[i, 57] = np.sum(test_dat[i, 13:26]*test_dat[i, 39:52])
    # for j in range(0,4):
    #     test_dat[i,52+j] = np.sum((test_dat[i,j*13:(j+1)*13]))
    # for j in range(0,13):
    #     test_dat[i,56+j] = test_dat[i,j] + test_dat[i,13+j] + test_dat[i,26+j] + test_dat[i,39+j]
time_taken2 = time.time() - since2
print(str(time_taken2) + " seconds taken for rearranging the data\n")

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
file.close()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, pred)
np.savetxt("cm.csv", cm, delimiter=",")

print "Confusion matrix saved in cm.csv"

num_data = 100

#new_dat = [None] * len(test_dat)
#new_dat = [None] * num_data
new_dat = []
#new_best = [None] * len(test_dat)
#new_best = [None] * num_data
new_best = []

#for i in range(len(test_dat)):

since = time.time()

for i in range(num_data):
    curr_best, curr_best_index = pred[i], i
    for j in np.where(test_dat[i][0:52] == 1)[0]: # where the card is present
        test_dat[i][j] = 0.0 # suppose that card is absent
        for k in np.where(test_dat[i][0:52] == 0)[0]: # and then suppose some other card is present
            if (k != j) and (test_dat[i][k] != 1.0):
                test_dat[i][k] = 1.0
                for l in range(0, 10, 2):
                    test_dat[i, 52] = np.sum(test_dat[i, 0:13]*test_dat[i, 13:26])
                    test_dat[i, 53] = np.sum(test_dat[i, 26:39]*test_dat[i, 39:52])
                    test_dat[i, 54] = np.sum(test_dat[i, 13:26]*test_dat[i, 26:39])
                    test_dat[i, 55] = np.sum(test_dat[i, 0:13]*test_dat[i, 26:39])
                    test_dat[i, 56] = np.sum(test_dat[i, 0:13]*test_dat[i, 39:52])
                    test_dat[i, 57] = np.sum(test_dat[i, 13:26]*test_dat[i, 39:52])
                x_in = np.array(test_dat[i]).reshape(1, 58)
                new_pred = rc.classify(x_in) # evaluate the hand of the combination with the supposed card
                if (new_pred > curr_best):
                    print("better combination found!\tnew hand: " + str(new_pred))
                    curr_best, curr_best_index = new_pred, i
                    #new_dat[i] = test_dat[i]
                    new_dat.append(test_dat[i])
                test_dat[i][k] = 0.0
        test_dat[i][j] = 1.0
    #new_best[i] = curr_best
    new_best.append(curr_best)
    print("new best: " + str(new_best[i]) + "\tprev best: " + str(pred[i]))
time_taken = time.time() - since
print (str(time_taken) + " seconds elapsed for " + str(num_data) + " data points\n")
        
file2 = open('output_task2.txt', 'w')
for i in range(len(new_dat)):
    file2.write(str(new_dat[i]) + '\n')
file2.close()
