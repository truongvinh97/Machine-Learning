"""
This file is for multiclass fashion-mnist classification using TensorFlow

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *
import time

# [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
def tf_softmax_placeholder():
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.compat.v1.placeholder(tf.float32, name='y')

    return x,y

# [TODO 2.8] Create weights (W) using TF variables
def tf_softmax_w_variable(w_shape):
    w = tf.Variable(np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape) , name = "w" , dtype=tf.float32)

    return w

# [TODO 2.8] Create an SGD optimizer
def tf_softmax_create_optimizer(learning_rate, cost):
    tf.compat.v1.disable_v2_behavior()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return optimizer

# [TODO 2.8] Compute losses and update weights here
def tf_softmax_update_weight(sess, optimizer, cost,x, y, train_x, train_y, val_x, val_y):
    sess.run(optimizer, feed_dict={x: train_x, y: train_y})
    train_loss = sess.run(cost, feed_dict={x: train_x, y: train_y})
    val_loss = sess.run(cost, feed_dict={x: val_x, y: val_y})

    return train_loss, val_loss

# [TODO 2.9] Create a feed-forward operator 
def tf_softmax_feed_forward(x, w):
    pred = tf.matmul(x, w)
    pred_max = tf.reduce_max(pred, 1, True)
    pred -= pred_max
    exp_pred = tf.exp(pred)
    pred = exp_pred / tf.reduce_sum(exp_pred, 1, True)

    return pred

# [TODO 2.10] Write the cost function
def tf_softmax_cost(pred, y):
    cost = -tf.reduce_mean(tf.reduce_sum(y*tf.math.log(pred), 1))

    return cost

# [TODO 2.11] Define your own stopping condition here 
def is_stop_training(all_val_loss):
    is_stopped = False
    num_val_increase = 0
    if (len(all_val_loss)<2):
        return False
    for i in range(len(all_val_loss)-1,0,-1):
        if all_val_loss[i] > all_val_loss[i-1]:
            num_val_increase += 1
        if num_val_increase >= 4:
            return True

    return is_stopped

if __name__ == "__main__":
    np.random.seed(2020)
    tf.random.set_seed(2020)
    from IPython.display import clear_output
    clear_output(wait=True)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
   
    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
    x, y = tf_softmax_placeholder()

    # [TODO 2.8] Create weights (W) using TF variables 
    w_shape = (train_x.shape[1],10)
    w = tf_softmax_w_variable(w_shape)

    # [TODO 2.9] Create a feed-forward operator 
    pred = tf_softmax_feed_forward(x, w)

    # [TODO 2.10] Write the cost function
    cost = tf_softmax_cost(pred, y) 

    # Define hyper-parameters and train-related parameters
    #num_epoch = 10000
    num_epoch = 3340
    learning_rate = 0.01    

    # [TODO 2.8] Create an SGD optimizer
    optimizer = tf_softmax_create_optimizer(learning_rate,cost)

    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf.compat.v1.global_variables_initializer()
    
    with tf.compat.v1.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.time()
            # [TODO 2.8] Compute losses and update weights here
            train_loss, val_loss = tf_softmax_update_weight(sess, optimizer, cost,x,y, train_x, train_y, val_x, val_y)

            train_loss = sess.run(cost, feed_dict={x:train_x, y:train_y}) 
            val_loss = sess.run(cost, feed_dict={x:val_x, y:val_y}) 
            # Update weights
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            # [TODO 2.11] Define your own stopping condition here 
            if is_stop_training(all_val_loss):
                break

            if (e % epochs_to_draw == epochs_to_draw-1):
                clear_output(wait=True)
                plot_loss(all_train_loss, all_val_loss)
                w_  = sess.run(w)
                draw_weight(w_)
                plt.show()
                plt.pause(0.1)     
                print("Epoch %d: train loss: %.5f || val loss: %.5f" % (e+1, train_loss, val_loss))

            toc = time.time()
            print(toc-tic)
          
        
        y_hat = sess.run(pred, feed_dict={x: test_x})
        np.set_printoptions(precision=2)
        confusion_mat = test(y_hat, test_y)
        print('Confusion matrix:')
        print(confusion_mat)
        print('Diagonal values:')
        print(confusion_mat.flatten()[0::11])
