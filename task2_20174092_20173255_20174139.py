import numpy as np
import tensorflow as tf 
import time 

# Read the test data
test_x = np.genfromtxt('test_data.csv', dtype=np.uint8, delimiter=',')
# test_y = np.genfromtxt('test_hand.csv', dtype=np.uint8, delimiter=',')

test_dat = np.zeros((len(test_x), 58))

for i in range(len(test_x)):
    for j in range(0,10,2):
        test_dat[i, (test_x[i,j]-1)*13 + (test_x[i,j+1]-1)]=1
    # include features 
    test_dat[i, 52] = np.sum(test_dat[i, 0:13]*test_dat[i, 13:26])
    test_dat[i, 53] = np.sum(test_dat[i, 26:39]*test_dat[i, 39:52])
    test_dat[i, 54] = np.sum(test_dat[i, 13:26]*test_dat[i, 26:39])
    test_dat[i, 55] = np.sum(test_dat[i, 0:13]*test_dat[i, 39:52])
    test_dat[i, 56] = np.sum(test_dat[i, 0:13]*test_dat[i, 26:39])
    test_dat[i, 57] = np.sum(test_dat[i, 13:26]*test_dat[i, 39:52])
    
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 58   # 4 * 13 cards + 6 features 
num_classes = 10 # number of classes 

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

# five hot code [0~51] and include features [52~57]
def five_hot_code(vector, num_cards=58):
    five_hots = np.zeros((1,num_cards))
    for i in vector:
        five_hots[0][i] = 1

    five_hots[0, 52] = np.sum(five_hots[0, 0:13]*five_hots[0, 13:26])
    five_hots[0, 53] = np.sum(five_hots[0, 26:39]*five_hots[0, 39:52])
    five_hots[0, 54] = np.sum(five_hots[0, 13:26]*five_hots[0, 26:39])
    five_hots[0, 55] = np.sum(five_hots[0, 0:13]*five_hots[0, 39:52])
    five_hots[0, 56] = np.sum(five_hots[0, 0:13]*five_hots[0, 26:39])
    five_hots[0, 57] = np.sum(five_hots[0, 13:26]*five_hots[0, 39:52])
    return five_hots

# Find the index that the value is equal to 1
def find_index(vector):
    list = []
    for i in range(len(vector)):
        if vector[i] == 1:
            list.append(i)
    return list

# make the matrix that includes all the possibilies about changing
# the cards in one test dataset
def make_testset(input):
    list = []
    index = find_index(input)
    for i in range(5):
        real_val = index[i]

        for j in range(52):
            index[i] = j
            encode = five_hot_code(index)
            if (sum(encode[0][0:52]) == 5):
                list.append(encode[0])

        index[i] = real_val
    return list 


# tf Graph input
X = tf.placeholder("float", [None, num_input])

# Learning rate 
lr = tf.placeholder(tf.float32)

# Construct model
logits = neural_net(X)

# Evaluate model (with test logits, for dropout to be disabled)
pred_classes = tf.argmax(logits, axis=1)

# Create saver
saver = tf.train.Saver()

with tf.Session() as sess:
    f = open('output_task2', 'w')
    # Restore pre-trained model 
    saver.restore(sess, "model/99/card_model-6000")
    start= time.time()
    for i in range(len(test_dat)):
        list = make_testset(test_dat[i])

        pred = sess.run(pred_classes, feed_dict={X: list})
    
        max_idx = np.argmax(pred)

        card_idx = find_index(list[max_idx][0:52])
        # write the output as csv file. 
        for j in range(len(card_idx)):
            f.write(str(card_idx[j]//13 + 1)+",")
            if j == len(card_idx)-1:
                f.write(str(card_idx[j]%13 + 1)+"\n")
            else:
                f.write(str(card_idx[j]%13 + 1)+",")

    
    f.close()
    end = time.time()
    print str(end-start) + 'seconds'
