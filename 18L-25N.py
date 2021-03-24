import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import prep as dp
import datetime

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
import os

# uncomment this block if you want to run on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)

# uncomment this if you want to seed the random generator
from numpy.random import seed #
seed(5)# keras seed fixing
import tensorflow as tf
tf.random.set_seed(5)# tensorflow seed fixing
print(tf.version.VERSION)


# We load the data we created earlier
#data_folder = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\"
load_train_label_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\train_labels_af6000.npy"
load_train_data_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\train_data_af6000.npy"
load_eval_label_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\eval_labels_af6000.npy"
load_eval_data_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\eval_data_af6000.npy"

x_train, x_eval = np.load(load_train_data_file), np.load(load_eval_data_file)
y_train, y_eval = np.load(load_train_label_file), np.load(load_eval_label_file)

print(x_eval.shape)
print(x_train.shape)

# dx_train=pd.DataFrame(data=x_train)
# dx_eval=pd.DataFrame(data=x_eval)
# dy_train=pd.DataFrame(data=y_train)
# dy_eval=pd.DataFrame(data=y_eval)
# sns.pairplot(dx_train,diag_kind='kde')
# plt.show()

# Create a normalization layer  (the input of the network should be numbers between -1 and 1)
normalizer = preprocessing.Normalization()
normalizer.adapt(x_train)  # adapts the normalization layer to the size of the training data

# Definition of the neural network
model = tf.keras.Sequential([  # sequential= the layers in the network are arranged in the order we type them
    normalizer,
    tf.keras.layers.Dense(25,
                          activation='selu'),
    # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    tf.keras.layers.Dense(25, activation='selu'),
    # # tf.keras.layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

model.summary()  # prints a summary of your model

# compile/create our network
learning_rate=0.001
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),  # optimizer=> the function that trains the network
    loss='mean_absolute_error',  # the function that evaluates the network. the optimizer tries to minimize the loss
    metrics=['mae', 'mse'])  # different metrics that are tracked during training

# training of the network inputs:
# x_train-> training data
# y_train -> training labels
# batch_size -> how many data should the network see before it updates the loss function (the higher the better, memory limited)
# epochs -> how many rounds of training should you have. 1 epochs= the networks trains on the whole dataset
# validation_data-> optional you can add the validation data(the network doesnt train on them) to evaluate the network during training
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1,mode='min',patience=150)
sm=tf.keras.callbacks.ModelCheckpoint('best_model_weights.h5', monitor='val_loss', mode='min',save_weights_only=True, save_best_only=True,verbose=1)

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + "learning_rate_"+str(learning_rate)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history1 = model.fit(x_train, y_train, batch_size=100, epochs=1000,
                     validation_data=(x_eval, y_eval),
                     callbacks= [tensorboard_callback, es, sm],
                     shuffle=False
                     )

dp.show_train_history(history1)  # the function we made in data_prep

# res=model.predict(x_eval)
# print(res)
# plt.scatter(res,y_eval)
# plt.show()
model.save("C:\\Users\\rtene\\PycharmProjects\\Neural_network\\model_af6000.h5")  #saves the model
model.save_weights("C:\\Users\\rtene\\PycharmProjects\\Neural_network\\model_weights_a6000_new.h5") #saves the model weights

model.evaluate(x_eval, y_eval)  # evaluate the model with ur validation data

# test_predictions = model.predict(x_eval)
#
# a = plt.axes(aspect='equal')
# plt.scatter(y_eval, test_predictions)
# plt.xlabel('True Values [um]')
# plt.ylabel('Predictions [um]')
# plt.show()

#tensorboard --logdir logs/fit



##
# tf.keras.layers.Dense(25, activation='selu'),
# # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# tf.keras.layers.Dense(25, activation='selu'),
# # tf.keras.layers.Dropout(0.2),
##
# tf.keras.layers.Dense(25, activation='relu'),
# # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# tf.keras.layers.Dense(25, activation='relu'),
# # tf.keras.layers.Dropout(0.2),

# tf.keras.layers.Dense(25, activation='sigmoid'),  # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# tf.keras.layers.Dense(25, activation='sigmoid'),
# # # tf.keras.layers.Dropout(0.2),

# tf.keras.layers.Dense(25,
#                       activation='tanh'),  # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# tf.keras.layers.Dense(25, activation='tanh'),
# # # tf.keras.layers.Dropout(0.2),

# tf.keras.layers.Dense(25,
#                       activation='elu'),  # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# tf.keras.layers.Dense(25, activation='elu'),
# # # tf.keras.layers.Dropout(0.2),

