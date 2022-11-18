"""
This file is for binary classification using TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_vehicle_data
from logistic_np import *

# [TODO 1.11] Create TF placeholders to feed train_x and train_y when training
def create_placeholders():
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.compat.v1.placeholder(tf.float32, name='y')

    return x, y

# [TODO 1.12] Create weights (W) using TF variables
def create_w_variable(w_shape):
    w = tf.Variable(np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape) , name = "w" , dtype=tf.float32)
 
    return w

# [TODO 1.13] Create a feed-forward operator
def tf_feed_forward(x, w):
    pred = 1.0 / (1.0 + tf.exp(-tf.matmul(x, w)))

    return pred

# [TODO 1.14] Write the cost function
def tf_cost(y, pred):
    cost = -tf.reduce_mean((y*tf.math.log(pred) + (1-y)*tf.math.log(1-pred)))

    return cost

# [TODO 1.15] Create an SGD optimizer
def tf_optimizer(learning_rate, cost):
    tf.compat.v1.disable_v2_behavior()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return optimizer

# [TODO 1.16] Compute loss and update weights here
def tf_update_weight(session, cost,x,y, optimizer, train_x, train_y):
    session.run(optimizer, feed_dict={x: train_x, y: train_y})
    loss = session.run(cost, feed_dict={x: train_x, y: train_y})

    return loss

if __name__ == "__main__":

    normalize_method = "all_pixel"

    np.random.seed(2018)
    tf.random.set_seed(2018)

    # Load data from file
    # Make sure that vehicles.dat is in data/
    train_x, train_y, test_x, test_y = get_vehicle_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    #generate_unit_testcase(train_x.copy(), train_y.copy())
    # logistic_unit_test()

    # Normalize our data: choose one of the two methods before training
    #train_x, test_x = normalize_all_pixel(train_x, test_x)
    if normalize_method == "all_pixel":
        train_x, test_x = normalize_all_pixel(train_x, test_x) 
    else:
        train_x, test_x = normalize_per_pixel(train_x, test_x) 

    # Reshape our data
    # train_x: shape=(2400, 64, 64) -> shape=(2400, 64*64)
    # test_x: shape=(600, 64, 64) -> shape=(600, 64*64)
    train_x = reshape2D(train_x)
    test_x = reshape2D(test_x)

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    test_x = add_one(test_x)

    # [TODO 1.11] Create TF placeholders to feed train_x and train_y when training
    x,y = create_placeholders()

    # [TODO 1.12] Create weights (W) using TF variables
    w_shape = (train_x.shape[1],1)
    w = create_w_variable(w_shape)

    # [TODO 1.13] Create a feed-forward operator
    pred = tf_feed_forward(x, w)

    # [TODO 1.14] Write the cost function
    #cost = -tf.reduce_sum(y*tf.log(pred)+(1-y)*tf.log(1-pred))/num_train
    cost = tf_cost(y, pred) 

    # Define hyper-parameters and train-related parameters
    num_epoch = 1000
    learning_rate = 0.01
#    momentum_rate = 0.9

    # [TODO 1.15] Create an SGD optimizer
    optimizer = tf_optimizer(learning_rate, cost)

    # Some meta parameters
    epochs_to_draw = 100
    all_loss = []
    plt.ion()

    # Start training
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.time()
            # [TODO 1.16] Compute loss and update weights here
            loss = tf_update_weight(sess, cost,x,y, optimizer, train_x, train_y)
            # Update weights...
            
            all_loss.append(loss)

            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_loss)
                plt.show()
                plt.pause(0.1)
                print("Epoch %d: loss is %.5f" % (e+1, loss))
            toc = time.time()
            print(toc-tic)
        y_hat = sess.run(pred, feed_dict={x: test_x})
        precision, recall, f1 = test(y_hat, test_y)
        print("Precision: %.3f" % precision)
        print("Recall: %.3f" % recall)
        print("F1-score: %.3f" % f1)
