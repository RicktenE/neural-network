import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import prep as dp

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Embedding
# from tqdm import tqdm
import shap

print(tf.__version__)

# reload the data

#data_folder = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\"
load_train_label_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\train_labels_af6000.npy"
load_train_data_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\train_data_af6000.npy"
load_eval_label_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\eval_labels_af6000.npy"
load_eval_data_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\eval_data_af6000.npy"

load_test_file = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\test_data_all_freq_6000.npy"

x_train = np.load(load_train_data_file)
y_train = np.load(load_train_label_file)
x_eval = np.load(load_eval_data_file)
y_eval = np.load(load_eval_label_file)
x_test = np.load(load_test_file)
# normalizer = preprocessing.Normalization(input_shape=[1,6000])
# normalizer.adapt(x_train)
wdw =500 # for old data
# wdw = 400
SIZE = wdw * 2  # Define the window size around the peaks as chosen in matlab

# remake our model
model = tf.keras.Sequential([
    preprocessing.Normalization(input_shape=[6*SIZE]),
    tf.keras.layers.Dense(25, activation='relu'),    # Dense=fully connected layer. 25= the number of neurons/nodes 'relu'=rectified linear unit activation function (standard activations)
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

model.summary()
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='mean_absolute_error',
    metrics=['accuracy'])

# we load the weights here
#model.load_weights("C:\\Users\\rtene\\PycharmProjects\\Neural_network\\model_weights_a6000_new.h5")
model.load_weights("D:\\Saxion\\Jaar 4\\Bachelor Thesis\\neural_network\\best_model_weights.h5")
# we use the trained network with the command model.predict . The results are saved on the output.  Flatten is used
# to create a single array at the end
test_predictions = model.predict(x_eval).flatten()
# test_predictions = model.predict(x_test).flatten()

print(test_predictions)
print(y_eval)
# various plots are below
a1 = plt.axes(aspect='equal')
plt.scatter(y_eval, test_predictions)
plt.xlabel('True Values [um]')
plt.ylabel('Predictions [um]')
plt.show()

# dswarm=pd.DataFrame(data=[y_eval,test_predictions])
# ax=sns.swarmplot(data=dswarm)

ax = sns.swarmplot(x=y_eval, y=test_predictions, size=2)
plt.xlabel('True Values [um]')
plt.ylabel('Predictions [um]')
plt.show()

ax = sns.boxplot(x=y_eval, y=test_predictions, whis=np.inf)
ax = sns.swarmplot(x=y_eval, y=test_predictions, size=3, color=".2")
plt.xlabel('True Values [um]')
plt.ylabel('Predictions [um]')
plt.show()

ax = sns.violinplot(x=y_eval, y=test_predictions, inner=None)
ax = sns.swarmplot(x=y_eval, y=test_predictions, size=3, color="white", edgecolor="gray")
plt.xlabel('True Values [um]')
plt.ylabel('Predictions [um]')
plt.show()

# ax=sns.violinplot(x=y_eval,y=test_predictions,inner=None)
# ax=sns.stripplot(x=y_eval,y=test_predictions,size=4,color="white",edgecolor="gray")
# plt.xlabel('True Values [um]')
# plt.ylabel('Predictions [um]')
# plt.show()

print(np.min(test_predictions))
a = 1
b = 1
c = 1
y5 = np.zeros((SIZE))
y6 = np.zeros((SIZE))
y7 = np.zeros((SIZE))
for i in range(len(y_eval)):
    if y_eval[i] == 5:
        y5[a] = test_predictions[i]
        a += 1
    elif y_eval[i] == 6:
        y6[a] = test_predictions[i]
        b += 1
    elif y_eval[i] == 7:
        y7[a] = test_predictions[i]
        c += 1

y5 = y5[y5 != 0]
y7 = y7[y7 != 0]
y6 = y6[y6 != 0]
print(y5)
print(y6)
print(y7)
plt.figure(figsize=(8, 6))
plt.hist((y5), bins=np.linspace(4.5, 5.5, 50), log=False, alpha=0.9)
plt.hist((y6), bins=np.linspace(5, 7, 50), alpha=0.9)
plt.hist((y7), bins=np.linspace(6, 8, 50), alpha=0.9)
plt.show()
print("5um std:" + str(np.std(y5)))
print("6um std:" + str(np.std(y6)))
print("7um std:" + str(np.std(y7)))
# print(y5)
# print(y6)
# print(y7)
#
# print(x_eval.shape)
# print(x_test.shape)
# for i in range(20):
#     plt.plot(x_eval[i,:])
#     plt.plot(x_test[770+i,:])
#     plt.show()

start = time.time()
test_predictions1 = model.predict(x_test).flatten()
end = time.time()
print("Counted " + str(x_test.shape[0]) + " particles in :" + str(end - start) + "sec")

for i in range(len(test_predictions1)):
    if test_predictions1[i] >= 8:
        test_predictions1[i] = 7
        print("corrected value")

plt.hist((test_predictions1), bins=100, alpha=0.5)
# print(test_predictions1)
plt.show()

# x_train=np.transpose(x_train)
# dx_train=pd.DataFrame(data=x_train)


###### this block is for the shapley values
shap.initjs()

t5 = []
t6 = []
t7 = []
for i in range(len(y_eval)):
    if y_eval[i] == 5:
        t5.append(x_eval[i, :])

    elif y_eval[i] == 6:
        t6.append(x_eval[i, :])
    elif y_eval[i] == 7:
        t7.append(x_eval[i, :])
# we create the unmixed data :)
eval5 = np.array(t5)  # eval5 contains all the data of 5um
eval6 = np.array(t6)  # eval6 contains all the data of 6um
eval7 = np.array(t7)  # eval7 contains all the data of 7um

background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
# A network explainer is created here using 200 data points
e = shap.DeepExplainer(model, x_train[:wdw, :])

# asking the shapley values for the 6um using the eval6
shap_values = e.shap_values(eval6)
# print(shap_values)
# plot the feature attributions
shap.summary_plot(shap_values, eval6, max_display=10, )

sval = np.array(shap_values)
print(sval.shape)
x=sval[0,0,:]
print(x.shape)


print(a.shape)
for i in range(len(a[:,1])):
    plt.plot(a[i,:])
plt.show()

# save the values in a txt so we analyze in matlab
# save_shap_file = "C:\\Users\\Papadimitriouv\\Documents\\ML\\regress\\shap\\all_freq6000\\6um.txt"
# np.savetxt(save_shap_file, a, delimiter=',')
# a1 = np.sum(np.abs(a), axis=0)
# print(a.shape)
plt.plot(a1)
plt.show()
a1 = np.sum(a, axis=0)
# print(a.shape)
plt.plot(a1)
plt.show()
# np.save(save_eval_label_file,y_eval)
# explainer = shap.KernelExplainer(f, dx_train.iloc[:50,:])
# shap_values = explainer.shap_values(x_train[299,:], nsamples=500)
# shap.force_plot(explainer.expected_value, shap_values, x_train[299,:])


