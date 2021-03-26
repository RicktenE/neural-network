import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import prep as dp

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
#
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# ## for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

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
# wdw =500 # for old data
wdw = 500
SIZE = wdw * 2  # Define the window size around the peaks as chosen in matlab
act_func = 'selu'
# remake our model
model = tf.keras.Sequential([
    preprocessing.Normalization(input_shape=[6*SIZE]),
    tf.keras.layers.Dense(100, activation=act_func),
    tf.keras.layers.Dense(25, activation=act_func),
    tf.keras.layers.Dense(10, activation=act_func),
    # # tf.keras.layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

# model.summary()
# model.compile(
#     optimizer=tf.optimizers.Adam(learning_rate=0.001),
#     loss='mean_absolute_error',
#     metrics=['accuracy'])

# we load the weights here
#model.load_weights("C:\\Users\\rtene\\PycharmProjects\\Neural_network\\model_weights_a6000_new.h5")
model.load_weights("D:\\Saxion\\Jaar 4\\Bachelor Thesis\\neural network\\best_model_weights.h5")
# we use the trained network with the command model.predict . The results are saved on the output.  Flatten is used
# to create a single array at the end
eval_predictions = model.predict(x_eval).flatten()
# print(type(eval_predictions))
# print(y_eval)

# print("check 1")
# # various plots are below
# a1 = plt.axes(aspect='equal')
# plt.scatter(y_eval, eval_predictions)
# plt.title("Network performance on evaluation data set")
# plt.xlabel('True Values 'r'[$\mu$m]')
# plt.ylabel('Predictions 'r'[$\mu$m]')
# plt.show()
#
# dswarm=pd.DataFrame(data=[y_eval,eval_predictions])
# ax=sns.swarmplot(data=dswarm)
# print("check 2 ")
# ax = sns.swarmplot(x=y_eval, y=eval_predictions, size=1)
# plt.title("Network performance on evaluation data set")
# plt.xlabel('True Values 'r'[$\mu$m]')
# plt.ylabel('Predictions 'r'[$\mu$m]')
# plt.show()
#
# # print("check 3 ")
# # ax = sns.boxplot(x=y_eval, y=eval_predictions, whis=np.inf)
# # ax = sns.swarmplot(x=y_eval, y=eval_predictions, size=0.9, color=".2")
# # plt.title("Network performance on evaluation data set")
# # plt.xlabel('True Values 'r'[$\mu$m]')
# # plt.ylabel('Predictions 'r'[$\mu$m]')
# # plt.show()
#
# print("check 4 ")
# ax = sns.violinplot(x=y_eval, y=eval_predictions, inner=None)
# ax = sns.swarmplot(x=y_eval, y=eval_predictions, size=1.1, color="white", edgecolor="gray")
# plt.title("Network performance on evaluation data set")
# plt.xlabel("True Values "r"[$\mu$m]")
# plt.ylabel("Predictions "r"[$\mu$m]")
# plt.show()
#
#
# print("check 4.2 ")
# ax=sns.violinplot(x=y_eval,y=eval_predictions,inner=None)
# ax=sns.stripplot(x=y_eval,y=eval_predictions,size=1.1,color="white",edgecolor="gray")
# plt.title("Network performance on evaluation data set")
# plt.xlabel('True Values 'r'[$\mu$m]')
# plt.ylabel('Predictions 'r'[$\mu$m]')
# plt.show()
# #
# # print(np.min(test_predictions))
# a = 1
# b = 1
# c = 1
# y5 = np.zeros((len(y_eval)))
# y6 = np.zeros((len(y_eval)))
# y7 = np.zeros((len(y_eval)))
# for i in range(len(y_eval)):
#     if y_eval[i] == 5:
#         y5[a] = eval_predictions[i]
#         a += 1
#     elif y_eval[i] == 6:
#         y6[b] = eval_predictions[i]
#         b += 1
#     elif y_eval[i] == 7:
#         y7[c] = eval_predictions[i]
#         c += 1
#
# y5 = y5[y5 != 0]
# y7 = y7[y7 != 0]
# y6 = y6[y6 != 0]
# # print(y5)
# # print(y6)
# # print(y7)
# #
# print("check 5 ")
# plt.figure(figsize=(8, 6))
# plt.hist((y5), bins=np.linspace(4.5, 6, 50), log=False, alpha=0.9)
# plt.hist((y6), bins=np.linspace(5, 7, 50), alpha=0.9)
# plt.hist((y7), bins=np.linspace(6, 8.5, 50), alpha=0.9)
# plt.title("Network performance on evaluation data set")
# plt.xlabel("particle size "r"[$\mu$m]")
# plt.ylabel("Count")
# plt.show()
# print("5um std  x_eval: " + str(np.std(y5)))
# print("6um std: x_eval: " + str(np.std(y6)))
# print("7um std: x_eval: " + str(np.std(y7)))
#
# print("Counted at check 5 x_eval --" + str(y5.shape[0] + y6.shape[0] + y7.shape[0]) + " particles " )

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
print("check 6")
start = time.time()
test_predictions_mix = model.predict(x_test).flatten()
# print(test_predictions1.shape)
end = time.time()
# print("Counted at check 6 x_test --" + str(x_test.shape[0]) + " particles in :" + str(end - start) + "sec")

cnt = 0
for i in range(len(test_predictions_mix)):
    if test_predictions_mix[i] >= 15:
        test_predictions_mix[i] = 7
        cnt = cnt+1
        print("corrected " + str(cnt)+" value(s)")


# plt.hist((test_predictions_mix), bins=100)
# plt.title("check 6 - mixed particles")
# plt.xlabel("particle size "r"[$\mu$m]")
# plt.ylabel("Count")
# plt.show()

#######################################
k = 0
d = 0
e = 0
f = 0
y45_mix = np.zeros((len(test_predictions_mix)))
y5_mix = np.zeros((len(test_predictions_mix)))
y6_mix = np.zeros((len(test_predictions_mix)))
y7_mix = np.zeros((len(test_predictions_mix)))
for i in range(len(test_predictions_mix)):
    if test_predictions_mix[i] <= 4.60:
        y45_mix[k] = test_predictions_mix[i]
        k += 1
    elif test_predictions_mix[i] > 4.60 and test_predictions_mix[i] <= 5.5:
        y5_mix[d] = test_predictions_mix[i]
        d += 1
    elif test_predictions_mix[i] > 5.5 and test_predictions_mix[i] <= 6.5:
        y6_mix[e] = test_predictions_mix[i]
        e += 1
    elif test_predictions_mix[i] >= 6.5:
        y7_mix[f] = test_predictions_mix[i]
        f += 1

y45_mix = y45_mix[y45_mix != 0]
y5_mix = y5_mix[y5_mix != 0]
y6_mix = y6_mix[y6_mix != 0]
y7_mix = y7_mix[y7_mix != 0]

plt.figure(figsize=(8, 6))
plt.hist((y45_mix), bins=np.linspace(4, 5, 50), alpha=0.9, label= '4.5 $\mu$m; s.dev:  ' + str(np.round(np.std(y45_mix),2)) + ' $\mu$m; mean: '+ str(np.round(np.mean(y45_mix),2)) + ' $ \mu$m; cnt: '+ str(y45_mix.shape[0]))
plt.hist((y5_mix), bins=np.linspace(4.5, 6, 50), alpha=0.9, label= '5 $\mu$m;; s.dev:  ' + str(np.round(np.std(y5_mix),2)) + ' $\mu$m; mean: '+ str(np.round(np.mean(y5_mix),2)) + ' $ \mu$m; cnt: '+ str(y5_mix.shape[0]))
plt.hist((y6_mix), bins=np.linspace(5, 7, 50),   alpha=0.9, label= '6 $\mu$m;; s.dev:  ' + str(np.round(np.std(y6_mix),2)) + ' $\mu$m; mean: ' + str(np.round(np.mean(y6_mix),2)) +' $ \mu$m; cnt: '+ str(y6_mix.shape[0]))
plt.hist((y7_mix), bins=np.linspace(6, 8.5, 50), alpha=0.9, label= '7 $\mu$m;; s.dev:  ' + str(np.round(np.std(y7_mix),2)) + ' $\mu$m; mean: ' + str(np.round(np.mean(y7_mix),2)) + ' $ \mu$m; cnt: '+ str(y7_mix.shape[0]))

#####################################################################################
#####################################################################################
###############################      plot            ################################
#####################################################################################
#####################################################################################
plt.suptitle(r"$\bf{Train}$: 4.5,5,(no 6),7 -- combined  " + r"$\bf{Test}$: mixed "r"$\mu$m -- 11-17""\n" + r"$\bf{Activation}$ = "+act_func +" "+ r"$\bf{ Network:}$ 100-25-10 ")
#+ r"$\bf{ Diameter: }$ Exact"


plt.xlabel("particle size "r"[$\mu$m]")
plt.ylabel("Count")
plt.legend()
# plt.legend((y5_mix[1], y6_mix[1], y7_mix[1]), ('std dev 5 um' + str(np.std(y5_mix)), 'std dev 6 um' + str(np.std(y6_mix)), 'std dev 7 um' + str(np.std(y7_mix))))
plt.show()

print("Counted at check 6.2 --" + str(y45_mix.shape[0] + y5_mix.shape[0] + y6_mix.shape[0] + y7_mix.shape[0]) + " particles " )

print("5um std  x_test: " + str(np.std(y5_mix)))
print("6um std: x_test: " + str(np.std(y6_mix)))
print("7um std: x_test: " + str(np.std(y7_mix)))
#########################################################################################
print("Count should be  " + str(test_predictions_mix.shape))
# print("shape of predictions on x_eval  " + str(test_predictions.shape))

# x_train=np.transpose(x_train)
# dx_train=pd.DataFrame(data=x_train)


# ###### this block is for the shapley values
#
# shap.initjs()
# t45 = []
# t5 = []
# t6 = []
# t7 = []
# for i in range(len(y_eval)):
#     if y_eval[i] == 4.5:
#         t45.append(x_eval[i, :])
#     elif y_eval[i] == 5:
#         t5.append(x_eval[i, :])
#     elif y_eval[i] == 6:
#         t6.append(x_eval[i, :])
#     elif y_eval[i] == 7:
#         t7.append(x_eval[i, :])
# # we create the unmixed data :)
# eval45 = np.array(t45)  # eval45 contains all the data of 4.5 um
# eval5 = np.array(t5)  # eval5 contains all the data of 5um
# eval6 = np.array(t6)  # eval6 contains all the data of 6um
# eval7 = np.array(t7)  # eval7 contains all the data of 7um
#
# background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
# print("type of background: " + str(type(background)))
# # print("shape of background: " + str((background.shape)))
# # A network explainer is created here using 200 data points
# e = shap.DeepExplainer(model, x_train[:200, :])
#
# # asking the shapley values for the 6um using the eval6
# shap_values = e.shap_values(eval6)
# print(shap_values)
# # plot the feature attributions
# shap.summary_plot(shap_values, eval6, max_display=10, )
#
# sval = np.array(shap_values)
# print(sval.shape)
# x=sval[0,0,:]
# print(x.shape)
#
#
# print(a.shape)
# for i in range(len(a[:,1])):
#     plt.plot(a[i,:])
# plt.show()
#
# # save the values in a txt so we analyze in matlab
# # save_shap_file = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\neural network\\shap\\all_freq6000\\6um.txt"
# # np.savetxt(save_shap_file, a, delimiter=',')
# # a1 = np.sum(np.abs(a), axis=0)
# # print(a.shape)
# # plt.plot(a1)
# # plt.show()
# # a1 = np.sum(a, axis=0)
# print(a.shape)
# plt.plot(a1)
# plt.show()
# np.save(save_eval_label_file,y_eval)
# explainer = shap.KernelExplainer(f, dx_train.iloc[:50,:])
# shap_values = explainer.shap_values(x_train[299,:], nsamples=500)
# shap.force_plot(explainer.expected_value, shap_values, x_train[299,:])



