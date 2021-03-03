import matplotlib
import PIL
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# add a comment
import os
from os import listdir
from os.path import isfile, join


def read_txt(filepath):  # reads a txt file line by line and returns the data in an array and the length of the array
    a = np.loadtxt(filepath)
    data = a  # choosing the second column
    l = np.shape(data)
    return l, data


def norm01(data):  # normalize data between 0 and 1
    res = (data - data.min()) / (data.max() - data.min())
    return res


# x_train=np.concatenate((x_train0,x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7),axis=0)
def make_training_set(data_folder, split_ratio):  # creates the data set required for training and evaluation
    read_label = data_folder + "x_data_5.txt"  # the file that contains the 5um data
    l, data = read_txt(read_label)
    x5 = np.transpose(data)
    y5 = np.ones((l[1])) * 5  # the labels for 5um

    read_label = data_folder + "x_data_6.txt"  # the file that contains the 6um data
    l, data = read_txt(read_label)
    x6 = np.transpose(data)
    y6 = np.ones((l[1])) * 6  # the labels for 6um

    read_label = data_folder + "x_data_7.txt"  # the file that contains the 7um data
    l, data = read_txt(read_label)
    x7 = np.transpose(data)
    y7 = np.ones((l[1])) * 7  # the labels for 7um

    #combine all data as prep for training
    x_train = np.concatenate((x5, x6, x7), axis=0)
    y_train = np.concatenate((y5, y6, y7), axis=0)

    # this block mixes the data
    a = x_train
    b = y_train
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    x_train = np.array(a)
    y_train = np.array(b)

    # this splits the dataset to evaluation and training data set. split_ratio-> ration of evaluation to train data. Number between 0 and 1
    s = int(np.round(len(y_train) * split_ratio))

    if split_ratio == 0:
        x_train = x_train[s:, :]
        y_train = y_train[s:]
        return x_train, y_train  # x=data y=labels
    else:
        x_eval = x_train[0:s, :]
        y_eval = y_train[0:s]
        x_train = x_train[s:, :]
        y_train = y_train[s:]
        return x_train, y_train, x_eval, y_eval


def show_train_history(history1):  # function for displaying the training results
    plt.plot(history1.history['mae'], label='mae')
    plt.plot(history1.history['mse'], label='mse')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history1.history['loss'], label='loss')
    plt.plot(history1.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()


#Prepping train and evaluation data
def make_training_set_allfreq_os(data_folder,
                                split_ratio):  # similar to training set but this one puts all frequencies next to each other
    read_label = data_folder + "x_data_5.txt" # Name of the data for 5 micrometer beads
    l, data = read_txt(read_label)
    # print(len(data))
    x5 = np.transpose(data) #for old data
    # x5 = data
    # print(len(x5))

    # Window around the peak
    wdw = 500 #for old data
    # wdw = 400
    SIZE = wdw*2  # Define the window size around the peaks as chosen in matlab
    leng = int(len(x5[:, SIZE]) / 6) # Size of the arrays depending on the window size chosen in matlab
    leng5 = leng
    # print(leng)
    xx5 = np.zeros((6*SIZE, leng))  # initialize an array of 6000 by the number of samples
    a = 0
    for i in range(0, l[1], 6):
        xx5[:, a] = np.concatenate((x5[i, :SIZE], x5[i + 1, :SIZE], x5[i + 2, :SIZE], x5[i + 3, :SIZE],
                                    x5[i + 4, :SIZE], x5[i + 5, :SIZE]),
                                   axis=0)  # put the first 6 data from the txt next to each other
        a += 1
    x5 = xx5
    y5 = np.ones(leng) * 5

    # similar for 6 and 7um
    read_label = data_folder + "x_data_6.txt"
    l, data = read_txt(read_label)
    x6 = np.transpose(data) #for old data
    # x6 = data
    leng = int(len(x6[:, SIZE]) / 6)
    leng6=leng
    # print(leng)
    xx6 = np.zeros((6*SIZE, leng))
    a = 0
    for i in range(0, l[1], 6):
       xx6[:, a] = np.concatenate((x6[i, :SIZE], x6[i + 1, :SIZE], x6[i + 2, :SIZE], x6[i + 3, :SIZE],
                                    x6[i + 4, :SIZE], x6[i + 5, :SIZE]),
                                   axis=0)
       a += 1
    x6 = xx6
    y6 = np.ones(leng) * 6
    read_label = data_folder + "x_data_7.txt"
    l, data = read_txt(read_label)
    x7 = np.transpose(data) #for old  data
    # x7 = data
    leng = int(len(x7[:, SIZE]) / 6)
    leng7=leng
    # print(leng)
    xx7 = np.zeros((6*SIZE, leng))
    a = 0
    for i in range(0, l[1], 6):
        xx7[:, a] = np.concatenate((x7[i, :SIZE], x7[i + 1, :SIZE], x7[i + 2, :SIZE], x7[i + 3, :SIZE],
                                    x7[i + 4, :SIZE], x7[i + 5, :SIZE]),
                                   axis=0)
        a += 1
    x7 = xx7
    y7 = np.ones(leng) * 7
    x5 = np.transpose(x5)
    x6 = np.transpose(x6)
    x7 = np.transpose(x7)
    s5 = int(np.round(len(y5) * split_ratio))
    s6 = int(np.round(len(y6) * split_ratio))
    s7 = int(np.round(len(y7) * split_ratio))
    if split_ratio == 0:
        x5 = x5[s5:, :]
        y5 = y5[s5:]
        x6 = x6[s6:, :]
        y6 = y6[s6:]
        x7 = x7[s7:, :]
        y7 = y7[s7:]
    else:
        x5_eval = x5[0:s5, :]
        y5_eval = y5[0:s5]
        x5 = x5[s5:, :]
        y5 = y5[s5:]
        x6_eval = x6[0:s6, :]
        y6_eval = y6[0:s6]
        x6 = x6[s6:, :]
        y6 = y6[s6:]
        x7_eval = x7[0:s7, :]
        y7_eval = y7[0:s7]
        x7 = x7[s7:, :]
        y7 = y7[s7:]
    # combine 5,6,7um data in one array
    # x5 = np.transpose(x5)
    # x6 = np.transpose(x6)
    # x7 = np.transpose(x7)

# ######################################################
#     # Making sure all data sets are equally large. If 2000 data points are trained on 6 mu, the 5.6 will be counted as a 6 sooner then when it is trained on equal footing
#     multiplier = 50
#     ids = np.arange(len(y5))
#     choices = np.random.choice(ids, multiplier*min(leng5,leng6,leng7))
#     xx5 = x5[choices]
#     yy5 = y5[choices]
#     x5 = xx5
#     y5=yy5
#     ids = np.arange(len(y6))
#     # print((min(leng5,leng6,leng7)))
#     # print(type(leng5))
#     choices = np.random.choice(ids, multiplier*min(leng5,leng6,leng7))
#     xx6 = x6[choices]
#     yy6 = y6[choices]
#     # print(x5.shape)
#     # print(x6.shape)
#     # print(xx6.shape)
#     y6 = yy6
#     x6 = xx6
#     ids = np.arange(len(y7))
#     choices = np.random.choice(ids, multiplier*min(leng5,leng6,leng7))
#     xx7 = x7[choices]
#     yy7 = y7[choices]
#     # print(x5.shape)
#     # print(x7.shape)
#     # print(xx7.shape)
#     x7 = xx7
#     y7 = yy7
################################################################################

    x5 = np.transpose(x5)
    x6 = np.transpose(x6)
    x7 = np.transpose(x7)

    #printing out the shape of the final data
    print("x5 shape" + str(x5.shape))
    print("x6 shape" + str(x6.shape))
    print("x7 shape" + str(x7.shape))


    x5_eval = np.transpose(x5_eval)
    x6_eval = np.transpose(x6_eval)
    x7_eval = np.transpose(x7_eval)
    x_train = np.concatenate((x5, x6, x7), axis=1)
    y_train = np.concatenate((y5, y6, y7), axis=0)
    x_eval = np.concatenate((x5_eval, x6_eval, x7_eval), axis=1)
    y_eval = np.concatenate((y5_eval, y6_eval, y7_eval), axis=0)
    x_train = np.transpose(x_train)
    x_eval = np.transpose(x_eval)
    print("x_train shape " +str(x_train.shape))
    print("y_train shape" +str(y_train.shape))
    # mixing the data
    a = x_train
    b = y_train
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    x_train = np.array(a)
    y_train = np.array(b)
    if split_ratio == 0:
        return x_train, y_train
    else:
        return x_train, y_train, x_eval, y_eval

###Prepping Test data

def prep_test_data_allfreq(data_folder_test):  # prepares the mixed beads data by putting frequencies next to each other
    read_label = data_folder_test + "x1_data_mix.txt"
    l, data = read_txt(read_label)
    x_mix = np.transpose(data) #for old data
    # x_mix = data
    wdw = 500  # for Old data
    # wdw =400
    SIZE = wdw * 2  # Define the window size around the peaks as chosen in matlab
    # print("l is " + str(l[1]))
    leng = int(len(x_mix[:, 1]) / 6)
    # print(leng)
    xx_mix = np.zeros((6*SIZE, leng))
    a = 0
    for i in range(0, l[1], 6):
        xx_mix[:, a] = np.concatenate(
            (x_mix[i, :SIZE], x_mix[i + 1, :SIZE], x_mix[i + 2, :SIZE], x_mix[i + 3, :SIZE], x_mix[i + 4, :SIZE], x_mix[i + 5, :SIZE]),
            axis=0)
        a += 1
        # print("a is " + str(a))
        # print("i is " + str(i))
    x = np.transpose(xx_mix)
    # x = xx_mix
    # print("x_mix shape" + str(x.shape))

    return x


def prep_test_data(data_folder_test):  # puts mixed data in one array (single frequency)
    read_label = data_folder_test + "x1_data_mix.txt"
    l, data = read_txt(read_label)
    x = np.transpose(data)

    return x

####################Train / evaluation  data
data_folder = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\"
#If using this data transpose everything and set window to 500 indicated as old data

#data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\11-06\\"
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\11-17\\"
#data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\12-07\\"
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\Combined_data\\"


# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\11-06\\"
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\11-17\\"
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\12-07\\"
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\Combined_data\\"

#Directly in the matlab file for window size influence investigation
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"


####################Test data

data_folder_test = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\"

# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\11-17\\"

# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\11-06\\"
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\11-17\\"
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\12-07\\"
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\baseline removed\\Combined_data\\"

#Directly in the matlab file for window size influence investigation
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"

x_test=prep_test_data_allfreq(data_folder_test)
# x_test=prep_test_data(data_folder_test)
print("test data shape" + str(x_test.shape))
save_test_file="C:\\Users\\rtene\\PycharmProjects\\Neural_network\\test_data_all_freq_6000.npy"
np.save(save_test_file,x_test)
# #
# #


################## Train data
x_train, y_train, x_eval, y_eval=make_training_set_allfreq_os(data_folder,0.15)
#
# print(np.shape(x_train))
#
# print((y_train))
#
# print(np.shape(x_eval))
#
# print(np.shape(y_eval))

# #save_train_label_file="D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\train_labels_af6000.npy"
save_train_label_file="C:\\Users\\rtene\\PycharmProjects\\Neural_network\\train_labels_af6000.npy"
# save_train_label_file=""D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\11-17\\train_labels_af6000.npy"
np.save(save_train_label_file,y_train)
# #save_train_data_file="D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\train_data_af6000.npy"
save_train_data_file="C:\\Users\\rtene\\PycharmProjects\\Neural_network\\train_data_af6000.npy"
# save_train_data_file=""D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Processed_data\\with baseline\\11-17\\train_labels_af6000.npy"
np.save(save_train_data_file,x_train)
#
#
# #
save_eval_label_file="C:\\Users\\rtene\\PycharmProjects\\Neural_network\\eval_labels_af6000.npy"
np.save(save_eval_label_file,y_eval)
save_eval_data_file="C:\\Users\\rtene\\PycharmProjects\\Neural_network\\eval_data_af6000.npy"
np.save(save_eval_data_file,x_eval)
#
#
# print(x_train.shape)
# print(x_eval.shape)

