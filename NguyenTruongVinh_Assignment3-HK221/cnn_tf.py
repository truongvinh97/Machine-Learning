import pandas as pd
from util import * 
import matplotlib.pyplot as plt


#Import Numpy for statistical calculations
import numpy as np

# Import Warnings 
import warnings
warnings.filterwarnings('ignore')

# Import matplotlib Library for data visualisation
import matplotlib.pyplot as plt

#Import train_test_split from scikit library
from sklearn.model_selection import train_test_split

# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow.compat.v1 as tf


def mnist_classification():
    # Load data from file
    # Make sure that fashion-mnist/*.gz is in data/
    x_train, y_train, x_validate, y_validate, x_test, y_test = get_mnist_data(1)

    x_train, x_validate, x_test = normalize(x_train, x_validate, x_test)    

    y_train = y_train.flatten().astype(np.int32)
    y_validate = y_validate.flatten().astype(np.int32)
    y_test = y_test.flatten().astype(np.int32)
    image_shape = (28,28,1)

    x_train = x_train.reshape(x_train.shape[0],*image_shape)
    x_test = x_test.reshape(x_test.shape[0],*image_shape)
    x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)

    # CNN parameters
    learning_rate = 0.001
    epochs_x = 50
    batch_size = 512

    # Choose activation function
    activation = tf.nn.relu

    # Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    cnn_model = Sequential([
                Conv2D(filters=32,kernel_size=3,activation=activation,input_shape = image_shape),
                MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
                Dropout(0.2),
                Flatten(), # flatten out the layers
                Dense(32,activation=activation),
                Dense(10,activation = 'softmax')
                ])
   
    # Specify that all features have real-value data
    cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=optimizer,metrics =['accuracy'])
    cnn_model.summary()

    tensorboard = TensorBoard(
            log_dir = r'logs\{}'.format('cnn_1layer'),
            write_graph = True,
            histogram_freq=1,
            write_images = True
            )
    
    history = cnn_model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs_x,
                verbose=1,
                validation_data=(x_validate,y_validate),
                callbacks = [tensorboard]
                )
    
    score = cnn_model.evaluate(x_test,y_test,verbose=0)
    print('Test Loss : {:.4f}'.format(score[0]))
    print('Test Accuracy : {:.4f}'.format(score[1]))

    # Plot data
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def bat_classification():
    # Load data from file
    # Make sure that bat.dat is in data/
    x_train, y_train, x_test, y_test = get_bat_data()
    x_train, _, x_test = normalize(x_train, x_train, x_test)    

    y_test  = y_test.flatten().astype(np.int32)
    y_train = y_train.flatten().astype(np.int32)
    num_class = (np.unique(y_train)).shape[0]

    image_shape = (x_train.shape[0],1)

    x_train = x_train.reshape(x_train.shape[0],*image_shape)
    x_test = x_test.reshape(x_test.shape[0],*image_shape)
 
    # CNN parameters
    learning_rate = 0.01
    epochs_x = 100
    batch_size = 512

    # Choose activation function
    activation = tf.nn.relu

    # Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    cnn_model = Sequential([
                Conv2D(filters=32,kernel_size=3,activation=activation,input_shape = image_shape),
                MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
                Dropout(0.2),
                Flatten(), # flatten out the layers
                Dense(32,activation=activation),
                Dense(10,activation = 'softmax')
                ])
   
    # Specify that all features have real-value data
    cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=optimizer,metrics =['accuracy'])
    cnn_model.summary()

    tensorboard = TensorBoard(
            log_dir = r'logs\{}'.format('cnn_1layer'),
            write_graph = True,
            histogram_freq=1,
            write_images = True
            )
    
    history = cnn_model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs_x,
                verbose=1,
                validation_data=(x_train,x_train),
                callbacks = [tensorboard]
                )
    
    score = cnn_model.evaluate(x_test,y_test,verbose=0)
    print('Test Loss : {:.4f}'.format(score[0]))
    print('Test Accuracy : {:.4f}'.format(score[1]))

    # Plot data
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(2017) 

    plt.ion()
    #bat_classification()
    mnist_classification()
