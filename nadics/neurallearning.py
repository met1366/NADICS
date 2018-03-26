import collections
from types import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

def encodeTarget(target):
    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return all_Y

def ffnn(train_X, test_X, train_y, test_y):
    train_y = encodeTarget(train_y)
    test_y = encodeTarget(test_y)

    x_size = train_X.shape[1]
    h_size = 256 # Number of hidden nodes
    y_size = train_y.shape[1]

    ###########################################################################
    #                                                                         #
    # PLACEHOLDERS                                                            #
    #                                                                         #
    # Here x and y_ aren't specific values. Rather, they are each a           #
    # placeholder -- a value that we'll input when we ask TensorFlow to run a #
    # computation.                                                            #
    #                                                                         #
    # The input images x will consist of a 2d tensor of floating point        #
    # numbers. Here we assign it a shape of [None, 784], where 784 is the     #
    # dimensionality of a single flattened 28 by 28 pixel MNIST image, and    #
    # None indicates that the first dimension, corresponding to the batch     #
    # size, can be of any size. The target output classes y_ will also        #
    # consist of a 2d tensor, where each row is a one-hot 10-dimensional      #
    # vector indicating which digit class (zero through nine) the             #
    # corresponding MNIST image belongs to.                                   #
    #                                                                         #
    # The shape argument to placeholder is optional, but it allows TensorFlow #
    # to automatically catch bugs stemming from inconsistent tensor shapes.   #
    #                                                                         #
    ###########################################################################
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    y_ = tf.placeholder(tf.float32, shape=[None, y_size])

    ###########################################################################
    #                                                                         #
    # WEIGHT INITIALIZATION                                                   #
    #                                                                         #
    # One should generally initialize weights with a small amount of noise    # 
    # for symmetry breaking, and to prevent 0 gradients. Since we're using    #
    # ReLU neurons, it is also good practice to initialize them with a        #
    # slightly positive initial bias to avoid "dead neurons". Instead of      #
    # doing this repeatedly while we build the model, let's create two handy  #
    # functions to do it for us.                                              #
    #                                                                         #
    ###########################################################################

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def forwardprop(X, w_1, w_2):
        h = tf.nn.sigmoid(tf.matmul(X, w_1))
        yhat = tf.matmul(h, w_2)
        return yhat
    
    ###########################################################################
    #                                                                         #
    # CONVOLUTION AND POOLING                                                 #
    #                                                                         #
    # TensorFlow also gives us a lot of flexibility in convolution and        #
    # pooling operations. How do we handle the boundaries? What is our stride #
    # size? Here, we're always going to choose the vanilla version.           #
    # Our convolutions uses a stride of one and are zero padded so that the   #
    # output is the same size as the input. Our pooling is plain old max      #
    # pooling over 2x2 blocks. To keep our code cleaner, let's also abstract  #
    # those operations into functions.                                        #
    #                                                                         #
    ###########################################################################
    #    
    #def conv2d(x, W):
    #    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #
    #def max_pool_2x2(x):
    #    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    #                          strides=[1, 2, 2, 1], padding='SAME')

    ###########################################################################
    #                                                                         #
    # FIRST CONVOLUTIONAL LAYER                                               #
    #                                                                         #
    # It will consist of convolution, followed by max pooling. The            #
    # convolution will compute 32 features for each 5x5 patch. Its weight     #
    # tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are #
    # the patch size, the next is the number of input channels, and the last  #
    # is the number of output channels. We will also have a bias vector with  #
    # a component for each output channel.                                    #
    #                                                                         #
    # To apply the layer, we first reshape x to a 4d tensor, with the second  #
    # and third dimensions corresponding to image width and height, and the   #
    # final dimension corresponding to the number of color channels.          #
    #                                                                         #
    # We then convolve x_image with the weight tensor, add the bias, apply    #
    # the ReLU function, and finally max pool. The max_pool_2x2 method will   #
    # reduce the image size to 14x14.                                         #
    #                                                                         #
    ###########################################################################
    #
    #W_conv1 = weight_variable([5, 5, 1, 32])
    #b_conv1 = bias_variable([32])
    #
    #x_image = tf.reshape(x, [-1,28,28,1])
    #
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)

    ###########################################################################
    #                                                                         #
    # SECOND CONVOLUTIONAL LAYER                                              #
    #                                                                         #
    # In order to build a deep network, we stack several layers of this type. #
    # The second layer will have 64 features for each 5x5 patch.              #
    #                                                                         #
    ###########################################################################
    #
    #W_conv2 = weight_variable([5, 5, 32, 64])
    #b_conv2 = bias_variable([64])
    #
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    ###########################################################################
    #                                                                         #
    # WEIGHT INITIALIZATION FOR FORWARD PROPATATION                           #
    #                                                                         #
    ###########################################################################
    W_forw1 = weight_variable((x_size, h_size))
    W_forw2 = weight_variable((h_size, y_size))

    ###########################################################################
    #                                                                         #
    # FORWARD PROPAGATION                                                     #
    #                                                                         #
    ###########################################################################

    keep_prob = tf.placeholder(tf.float32)
    f1_drop = tf.nn.dropout(X, keep_prob)

    yhat = forwardprop(f1_drop, W_forw1, W_forw2)
    predict = tf.argmax(yhat, axis=1)

    ###########################################################################
    #                                                                         #
    # DENSELY CONNECTED LAYER                                                 #
    #                                                                         #
    # Now that the image size has been reduced to 7x7, we add a fully-        #
    # connected layer with 1024 neurons to allow processing on the entire     #
    # image. We reshape the tensor from the pooling layer into a batch of     #
    # vectors, multiply by a weight matrix, add a bias, and apply a ReLU.     #
    #                                                                         #
    ###########################################################################
    #
    #W_fc1 = weight_variable([7 * 7 * 64, 1024])
    #b_fc1 = bias_variable([1024])
    #
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    ###########################################################################
    #                                                                         #
    # DROPOUT                                                                 #
    #                                                                         #
    # To reduce overfitting, we will apply dropout before the readout layer.  #
    # We create a placeholder for the probability that a neuron's output is   #
    # kept during dropout. This allows us to turn dropout on during training, #
    # and turn it off during testing. TensorFlow's tf.nn.dropout op           #
    # automatically handles scaling neuron outputs in addition to masking     #
    # them, so dropout just works without any additional scaling.             #
    #                                                                         #
    ###########################################################################
    #
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    ###########################################################################
    #                                                                         #
    # READOUT LAYER                                                           #
    #                                                                         #
    # Finally, we add a layer, just like for the one layer softmax regression #
    # above.                                                                  #
    #                                                                         #
    ###########################################################################
    #
    #W_fc2 = weight_variable([1024, 10])
    #b_fc2 = bias_variable([10])
    #
    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ###########################################################################
    #                                                                         #
    # TRAIN AND EVALUATE THE MODEL                                            #
    #                                                                         #
    # - We will replace the steepest gradient descent optimizer with the more #
    # sophisticated ADAM optimizer.                                           #
    #                                                                         #
    # The keep_prob value is used to control the dropout rate used when       #
    # training the neural network. Essentially, it means that each connection #
    # between layers (in this case between the last densely connected layer   #
    # and the readout layer) will only be used with probability 0.5 when      #
    # training. This reduces overfitting. For more information on the theory  #
    # of dropout, you can see the original paper by Srivastava et al. To see  #
    # how to use it in TensorFlow, see the documentation on the               #
    # tf.nn.dropout() operator.                                               #
    #                                                                         #
    ###########################################################################
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_,
        logits=yhat))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # NOTE: The method initialize_all_variables() is deprecated, so we use 
    # tf.global_variables_initializer() instead!
    sess = tf.InteractiveSession(config=tf.ConfigProto(
        allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(30):
        if i % 10 == 0:
            print "==========\nEpoch: {0}\n==========".format(i)
        for i in range(len(train_X)):
            # TODO BATCHES
            sess.run(train_step, feed_dict={
                X: train_X[i: i + 1],
                y_: train_y[i: i + 1],
                keep_prob: 0.5})

    train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                             sess.run(predict, feed_dict={X: train_X, 
                                                          y_: train_y,
                                                          keep_prob: 1}))
    test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                             sess.run(predict, feed_dict={X: test_X, 
                                                          y_: test_y,
                                                          keep_prob: 1}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%\n"
          % (i + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()
