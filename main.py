# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import seaborn as sns
# import datetime
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import preprocessing
# import shap
# import time
np.set_printoptions(precision=3, suppress=True)

# from prep import make_training_set_allfreq_os, prep_test_data_allfreq
# from training import show_train_history, train
# from testing import test


#####################################################################################
##################### configure the setup ###########################################
#####################################################################################
wdw = 500                       # window size as chosen in matlab

#configuring the train data
date = 0                        # selecting the training data, 0 = combined
corrected_values = False        # True for using the corrected diameters False for using exact diameters
split_ratio = 0.15              # The percentage of validation data from the trianing set
exclude_size = 5                # 0 to exclude nothing, 45 for 4.5, 5 for 5, 6 for 6, 7 for 7
# exclude_day = 0                 # 0 to exclude nothing

#configuring the test data
date_test = 1117                # selecting the test data
test_size = 5                   # selecting the test size 0= mixed

#configuring the network
act_func = 'selu'               # Activation function used in training and testing
learning_rate = 0.001           # the learning rate of the training
patience = 300                  # the amount of epochs the training has to continue after finding the lowest val_loss
epochs = 2000                   # max number of epochs before the training stops

#What to run:
train_ = True # True if you want to train the network again
test_ = True # True if you want to test

#####################################################################################
##################### calling the functions ###########################################
#####################################################################################

from prep import make_training_set_allfreq_os
from prep import prep_test_data_allfreq
from training import train
from testing import test

if  train_ == True and test_ == True:
    make_training_set_allfreq_os(wdw,corrected_values,date,exclude_size,split_ratio)
    prep_test_data_allfreq(wdw,date_test,test_size)
    train(wdw,act_func,learning_rate,patience,epochs)
    test(wdw,act_func,date,exclude_size,date_test,test_size)

elif train_ != True and test_ == True:
    prep_test_data_allfreq(wdw,date_test,test_size)
    test(wdw,act_func,date,exclude_size,date_test,test_size)

elif train_ == True and test_ != True:
    make_training_set_allfreq_os(wdw,corrected_values,date,exclude_size,split_ratio)
    train(wdw,act_func,learning_rate,patience,epochs)


