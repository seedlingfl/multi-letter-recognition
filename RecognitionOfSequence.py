# coding utf-8

# # I. Data Aquisition
#  
# Before running this program, please first use 1notMNIST.ipynb to download and preprocess the notMNIST dataset.
# This program generates synthetic sequences of letters from the single-letter images in notMNIST dataset. 
# The sequence is limited to up to five letters. For sequences shorter than 5 letters, "blank" areas will be added to the end.


# These are all the modules used. 
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range



pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset1 = save['train_dataset']
  train_labels1 = save['train_labels']
  valid_dataset1 = save['valid_dataset']
  valid_labels1 = save['valid_labels']
  test_dataset1 = save['test_dataset']
  test_labels1 = save['test_labels']
  del save  # to help free up memory
  print('Training set', train_dataset1.shape, train_labels1.shape) # training set contains 500,000 single-letter images
  print('Validation set', valid_dataset1.shape, valid_labels1.shape) # validation set contains 10,000 images
  print('Test set', test_dataset1.shape, test_labels1.shape) # test set contains 10,000 images


N = 5 # maximum number of digits in a sequence
L = N + 1
img_size1 = 28
img_size2 = N * 28


# concat several (1-5) single-letter images of size 28*28 to a larger image of size 28*140
# method: for each single letter x in the dataset, choose two random integers n1 and n2 with values between 1 and 5, 
# then concat the next n1 letters and last n2 letters of x (inclusive) in the dataset. This method will generate 1,000,000 sequences for training. 
def generate_sequence_dataset(dataset,label):
    nb_rows = dataset.shape[0]
    print("the number of rows is",nb_rows);
    seq_dataset = np.zeros((2*nb_rows, img_size1, img_size2),dtype=np.float32)
    seq_labels = np.zeros((2*nb_rows,L), dtype=np.int32)
    for i in range(nb_rows):
        length = np.random.randint(1,N+1)
        if (i+length)>nb_rows:
            length=nb_rows-i
        # new_image = np.zeros((img_size1, img_size2),dtype=np.float32) # use value zero in blank areas
        new_image = 1.0/10*np.random.randn(img_size1, img_size2).astype(np.float32) # use Gaussian noise in blank areas
        new_label = np.zeros((L),dtype=np.int32)
        new_label[0] = length
        for j in range(length):
            new_image[0:img_size1, img_size1*j:img_size1*(j+1)] = dataset[i+j,:,:]
            new_label[j+1] = label[i+j]
        seq_dataset[2*i,:,:] = new_image
        seq_labels[2*i,:] = new_label
		
        length = np.random.randint(1,N+1)
        if (i+1-length)<0:
            length=i+1
        #new_image = np.zeros((img_size1, img_size2),dtype=np.float32) # use value zero in blank areas
        new_image = 1.0/10*np.random.randn(img_size1, img_size2).astype(np.float32) #use Gaussian noise in blank areas
        new_label = np.zeros((L),dtype=np.int32)
        new_label[0] = length
        for j in range(length):
            new_image[0:img_size1, img_size1*j:img_size1*(j+1)] = dataset[i-j,:,:]
            new_label[j+1] = label[i-j]
        seq_dataset[2*i+1,:,:] = new_image
        seq_labels[2*i+1,:] = new_label

    return seq_dataset, seq_labels

train_dataset, train_labels = generate_sequence_dataset(train_dataset1, train_labels1)
valid_dataset, valid_labels = generate_sequence_dataset(valid_dataset1, valid_labels1)
test_dataset, test_labels = generate_sequence_dataset(test_dataset1, test_labels1)

print('After concatting the digits:')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



# Reformat the sequence image into a TensorFlow-friendly shape:
# - convolutional layers need the image data formatted as a cube (width by height by #channels)
# - labels are float 1-hot encodings.

num_labels = 10 # for a, b, c, ..., i, j
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, img_size1, img_size2, num_channels)).astype(np.float32)
  num_rows = len(labels)
  newlabels = np.zeros((num_rows, L, num_labels))
  for i in range(num_rows):
      newlabels[i] = (np.arange(num_labels) == labels[i,:,None]).astype(np.float32)
  return dataset, newlabels

train_dataset, onehot_train_labels = reformat(train_dataset, train_labels)
valid_dataset, onehot_valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, onehot_test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, onehot_train_labels.shape)
print('Validation set', valid_dataset.shape, onehot_valid_labels.shape)
print('Test set', test_dataset.shape, onehot_test_labels.shape)


# I use small validation and test datasets for speed consideration.

small=1000
valid_dataset_small = valid_dataset[0:small,...]
valid_labels_small = valid_labels[0:small,...]
onehot_valid_labels_small = onehot_valid_labels[0:small,...]
small=2000
test_dataset_small = test_dataset[0:small,...]
onehot_test_labels_small = onehot_test_labels[0:small,...]




# # II. Build the Model and Train
#
# Architecture of the neural network: three convolutional layers, two fully connected layers and one output layer.
# The output layer consists of six softmax classifiers, one for the sequence length and the other five for each digit in the sequence.

batch_size = 20 # mini-batch size in each training step
patch_size = 5 # patch size for all the convolutional layers
depth = 48 # the number of chanels for the convolutional layers
num_hidden = 1000 # size of the fully-connected layers

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, img_size1, img_size2, num_channels))
  tf_train_labels_onehot = tf.placeholder(tf.float32, shape=(batch_size, L, num_labels))
  tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, L))
  tf_valid_dataset = tf.constant(valid_dataset_small)
  tf_test_dataset = tf.constant(test_dataset_small)
  keep_prob = tf.placeholder(tf.float32)
  

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth])) # 1st convolutional layer

  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth])) # 2nd convolutional layer

  layer2a_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2a_biases = tf.Variable(tf.constant(1.0, shape=[depth])) # 3rd convolutional layer

  layer3_weights = tf.Variable(tf.truncated_normal(
      [img_size1 // 4 * img_size2 // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden])) # fully connected layer
  
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_hidden], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden])) # fully connected layer

  len_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, L], stddev=0.1))
  len_biases = tf.Variable(tf.constant(1.0, shape=[L]))
  final_weights = tf.Variable(tf.truncated_normal(
      [N,num_hidden, num_labels], stddev=0.1))
  final_biases = tf.Variable(tf.constant(1.0, shape=[N,num_labels])) # output layer
  
  global_step = tf.Variable(0)  # count the number of steps taken.
  

  # Model.
  def ExtractFeature(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') # first convnet
    hidden = tf.nn.relu(conv + layer1_biases)
    hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # max pooling

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')# second convnet
    hidden = tf.nn.relu(conv + layer2_biases)
    hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # max pooling
    hidden = tf.nn.dropout(hidden, keep_prob) # dropout

    conv = tf.nn.conv2d(hidden, layer2a_weights, [1, 1, 1, 1], padding='SAME')# third convnet
    hidden = tf.nn.relu(conv + layer2a_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    reshape = tf.nn.dropout(reshape, keep_prob) # dropout

    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases) # fully connected layer
    hidden = tf.nn.dropout(hidden, keep_prob) # dropout
    
    return tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases) # 2nd fully connected layer
  
  def cross_entropy_with_logits(logits, labels):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    
  def loss_function(H): # H.size=num_batch*num_hidden
    loss = 0.0
    for i in range(batch_size):
        Hbatch = tf.pack([H[i], H[i], H[i], H[i], H[i]]) # N*num_hidden
        length = tf_train_labels[i,0]
        indices = tf.range(length)
        digit_weights = tf.gather(final_weights, indices) # len*num_hidden*num_labels
        digit_biases = tf.gather(final_biases, indices) # len*num_labels
        Hbatch2 = tf.gather(Hbatch, indices) # len*num_hidden
        Hbatch2 = tf.reshape(Hbatch2, [-1,1,num_hidden]) # len*1*num_hidden
       
        logits = tf.batch_matmul(Hbatch2, digit_weights) # len*1*num_labels
        logits = tf.reshape(logits, [-1,num_labels]) + digit_biases # len*num_labels
        loss += cross_entropy_with_logits(logits, tf.gather(tf_train_labels_onehot[i,1:,:], indices))          
    
    logits = tf.matmul(H, len_weights) + len_biases # num_batch*L
    loss += cross_entropy_with_logits(logits, tf_train_labels_onehot[:,0,0:L])
    
    return loss/batch_size
   
  def prediction(H):
    lenlabel = tf.nn.softmax(tf.matmul(H, len_weights) + len_biases) # num_batch*L
    label0 = tf.nn.softmax(tf.matmul(H, final_weights[0]) + final_biases[0]) # num_batch*num_labels
    label1 = tf.nn.softmax(tf.matmul(H, final_weights[1]) + final_biases[1])
    label2 = tf.nn.softmax(tf.matmul(H, final_weights[2]) + final_biases[2])
    label3 = tf.nn.softmax(tf.matmul(H, final_weights[3]) + final_biases[3])
    label4 = tf.nn.softmax(tf.matmul(H, final_weights[4]) + final_biases[4]) 
    digitlabel = tf.pack([label0, label1, label2, label3, label4], axis=1) # num_batch*N*num_labels
    return lenlabel, digitlabel
    
  # Training computation.
  H = ExtractFeature(tf_train_dataset)
  loss = loss_function(H)

  learning_rate = tf.train.exponential_decay(0.002, global_step,2000,0.99)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction_len, train_prediction_digit = prediction(ExtractFeature(tf_train_dataset))
  valid_prediction_len, valid_prediction_digit = prediction(ExtractFeature(tf_valid_dataset))
  test_prediction_len, test_prediction_digit = prediction(ExtractFeature(tf_test_dataset))



# calculate the accuracy: define a prediction as correct only if the predicted sequence is exactly the same with the true label
# the predicted sequence for an input image X is the sequence with maximum log-probability -- log[P(L|X)] + sum{log[P(S_i|X)]} 
# where L represents length, S_i represents the i-th digit, and i = 1,2,...,L
def accuracy(pred_len, pred_digit, labels):

    sum = 0
    
    for i in range(pred_len.shape[0]):# num_batch
        length = np.argmax(labels[i,0]) # true length
        
        flag=1
        prob=0
        maxprob=-10e8
        maxlength=0
        for j in range(N):
            if (j < length and np.argmax(pred_digit[i,j]) != np.argmax(labels[i,j+1])):
                flag = 0
                break
            else:
                prop = prob + np.log(np.max(pred_digit[i,j]))
                curprob = prob + np.log(pred_len[i,j+1])
                if(curprob > maxprob):
                    maxprob = curprob
                    maxlength = j+1

        if (maxlength != length):
            flag = 0
        
        if flag == 1:
            sum += 1
    return (100.0 * sum)/ pred_len.shape[0]


# This program will run for at least several days in a laptop for the num of steps below
num_steps = 400001


# Train the model using the synthetic dataset.
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  print('batch size:',batch_size)
  print('patch(kernel) size:',patch_size)
  print('depth = %d, num_hidden = %d' %(depth, num_hidden))
  print('keep_probability = %lf' %(0.9));

  accumulate_accuracy=0.0
  count=0
  flag=0
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    batch_labels_onehot = onehot_train_labels[offset:(offset + batch_size), :, :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, 
                 tf_train_labels_onehot : batch_labels_onehot, keep_prob: 0.9}
    _, l = session.run(
        [optimizer, loss], feed_dict=feed_dict)

    # accumulate the training error when flag == 1 (for 500 steps before every 20000 steps)
    if(flag==1):
      feed_dict={keep_prob:1.0, tf_train_dataset: batch_data}
      mini_accuracy = accuracy(train_prediction_len.eval(feed_dict=feed_dict),
                                                    train_prediction_digit.eval(feed_dict=feed_dict), batch_labels_onehot)
      accumulate_accuracy = (accumulate_accuracy*count + mini_accuracy)/(count+1.0)
      count+=1

    if (step % 1000 == 0): # print the mini-batch loss and accuracy at every 1000 steps
      print('Learning Rate = %f' %(learning_rate.eval()))
      print('Minibatch loss at step %d: %f' % (step, l))
      feed_dict={keep_prob:1.0, tf_train_dataset: batch_data}
      mini_accuracy = accuracy(train_prediction_len.eval(feed_dict=feed_dict),
			  train_prediction_digit.eval(feed_dict=feed_dict), batch_labels_onehot)
      print('Minibatch accuracy: %.1f%%' % mini_accuracy)

      if(flag==1):
        print('Training total accuracy: %.1f%%' % accumulate_accuracy) 
        flag=0

      valid_pred1, valid_pred2 = valid_prediction_len.eval(feed_dict={keep_prob:1.0}), valid_prediction_digit.eval(feed_dict={keep_prob:1.0})
      print('Validation accuracy: %.1f%%' % accuracy(valid_pred1, valid_pred2, onehot_valid_labels_small))
    
    if((step+500) % 20000 == 0): # accumulate the training error for 500 steps before printing at every 20000 steps
      count=0
      accumulate_accuracy=0
      flag=1
    if(step % 20000 == 0 and not step == 0): # calculate and print the test error every 20000 steps
      print('Test accuracy: %.1f%%' % accuracy(test_prediction_len.eval(feed_dict={keep_prob:1.0}), 
                                           test_prediction_digit.eval(feed_dict={keep_prob:1.0}), onehot_test_labels_small))





