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




#####################################################################################
##################### Preparing training and evaluation data ########################
#####################################################################################
#####################################################################################
#####################################################################################
def make_training_set_allfreq_os(wdw,corrected_values,date,exclude_size,split_ratio):  # similar to training set but this one puts all frequencies next to each other

 ####Initiating the function with some variable choices.
    # wdw = 500  # Window around the peak, has to be the same as chosen in the pre processing matlab file
    SIZE = wdw*2  # Define the window size around the peaks as chosen in matlab
    # corrected_values = True  #To chose this option is to chose to train the network with the corrected diameter as measured in Douwe's experiment.
    # exclude_size = 0  # Here you can choose which particles you want to exclude from the training set to check inter or extrapolation e.g. exclude = 5
    # date = 0 #  Choose date = 0 for combined data otherwise enter date. e.g. 6 nov = 1106
 #####################################################################################
 #####################################################################################
 #####################################################################################
 #####################################################################################
 #####################################################################################

########################################################################
    if date == 1106:
        data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
    elif date == 1117:
        data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
    elif date == 1207:
        data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
    elif date == 0:
        data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"

########################################################################
    #collecting data of 4.5 um beads
    if date == 0 or date == 1106:
        read_label = data_folder + "x_data_45.txt" # Name of the data for 4.5 micrometer beads
        l, data = read_txt(read_label)
        # print(len(data))
        x45 = np.transpose(data)
        # x45 = data
        # print("shape x45 before concatenate" + str(x45.shape))
        leng45 = int(len(x45[:, SIZE]) / 6) # Size of the arrays depending on the window size chosen in matlab
        leng45 = leng45
        # print("variable leng45: "+ str(leng45))
        xx45 = np.zeros((6*SIZE, leng45))  # initialize an array of 6000 by the number of samples
        a = 0
        for i in range(0, l[1], 6):
            xx45[:, a] = np.concatenate((x45[i, :SIZE], x45[i + 1, :SIZE], x45[i + 2, :SIZE], x45[i + 3, :SIZE],
                                        x45[i + 4, :SIZE], x45[i + 5, :SIZE]),
                                       axis=0)  # put the first 6 data from the txt next to each other
            a += 1
        x45 = xx45
        # print("shape x45 after concatenate: "+str(x45.shape))
        x45 = np.transpose(x45)

        if corrected_values == True:
            if date == 1106:
                read_label2 = data_folder + "D45_cor_11-06.txt"
            elif date == 0:
                read_label2 = data_folder + "D45_cor_combined.txt"
            l, data2 = read_txt(read_label2)
            y45 = data2
            # print("shape y45 " + str(y45.shape))
        else:
            y45 = np.ones(leng45) * 4.5

    #collecting data of 5 um beads
    if date != 1106:
        read_label = data_folder + "x_data_5.txt" # Name of the data for 5 micrometer beads
        l, data = read_txt(read_label)
        # print(len(data))
        x5 = np.transpose(data)
        # x5 = data
        # print("shape x5 before concatenate" + str(x5.shape))
        leng5 = int(len(x5[:, SIZE]) / 6) # Size of the arrays depending on the window size chosen in matlab
        # leng5 = leng
        # print("variable leng5: "+ str(leng5))
        xx5 = np.zeros((6*SIZE, leng5))  # initialize an array of 6000 by the number of samples
        a = 0
        for i in range(0, l[1], 6):
            xx5[:, a] = np.concatenate((x5[i, :SIZE], x5[i + 1, :SIZE], x5[i + 2, :SIZE], x5[i + 3, :SIZE],
                                        x5[i + 4, :SIZE], x5[i + 5, :SIZE]),
                                       axis=0)  # put the first 6 data from the txt next to each other
            a += 1
        x5 = xx5
        # print("shape x5 after concatenate: "+str(x5.shape))
        x5 = np.transpose(x5)

        if corrected_values == True:
            if date == 1117:
                read_label2 = data_folder + "D5_cor_11-17.txt"
            elif date == 1207:
                read_label2 = data_folder + "D5_cor_12-07.txt"
            elif date == 0:
                read_label2 = data_folder + "D5_cor_combined.txt"
            l, data2 = read_txt(read_label2)
            y5 = data2
            # print("shape y5 " + str(y5.shape))
        else:
            y5 = np.ones(leng5) * 5

    #collecting data of 6 um beads. 6um beads are in all data sets, therefore the if statement is more for esthetic reasons for an overall organisation of the file
    if date != 123456789:
        read_label = data_folder + "x_data_6.txt"
        l, data = read_txt(read_label)
        if date != 1106:
            x6 = np.transpose(data)
        else:
            x6 = data
        leng6 = int(len(x6[:, SIZE]) / 6)
        # print(leng6)
        xx6 = np.zeros((6*SIZE, leng6))
        a = 0
        for i in range(0, l[1], 6):
           xx6[:, a] = np.concatenate((x6[i, :SIZE], x6[i + 1, :SIZE], x6[i + 2, :SIZE], x6[i + 3, :SIZE],
                                        x6[i + 4, :SIZE], x6[i + 5, :SIZE]),
                                       axis=0)
           a += 1
        x6 = xx6
        # print("shape x6 after concatenate: " + str(x6.shape))
        x6 = np.transpose(x6)

        if corrected_values == True:
            if date == 1106:
                read_label2 = data_folder + "D6_cor_11-06.txt"
            elif date == 1117:
                read_label2 = data_folder + "D6_cor_11-17.txt"
            elif date == 1207:
                read_label2 = data_folder + "D6_cor_12-07.txt"
            elif date == 0:
                read_label2 = data_folder + "D6_cor_combined.txt"
            l, data2 = read_txt(read_label2)
            y6 = data2
            # print("y6 shape " + str(y6.shape))
        else:
            y6 = np.ones(leng6) * 6

    #collecting the 7um data
    if date != 1106:
        read_label = data_folder + "x_data_7.txt"
        l, data = read_txt(read_label)
        x7 = np.transpose(data)
        # x7 = data
        leng7 = int(len(x7[:, SIZE]) / 6)
        # print(leng7)
        xx7 = np.zeros((6*SIZE, leng7))
        a = 0
        for i in range(0, l[1], 6):
            xx7[:, a] = np.concatenate((x7[i, :SIZE], x7[i + 1, :SIZE], x7[i + 2, :SIZE], x7[i + 3, :SIZE],
                                        x7[i + 4, :SIZE], x7[i + 5, :SIZE]),
                                       axis=0)
            a += 1
        x7 = xx7
        # print("shape x7 after concatenate: "+str(x7.shape))
        x7 = np.transpose(x7)

        if corrected_values == True:
            if date == 1117:
                read_label2 = data_folder + "D7_cor_11-17.txt"
            elif date == 1207:
                read_label2 = data_folder + "D7_cor_12-07.txt"
            elif date == 0:
                read_label2 = data_folder + "D7_cor_combined.txt"
            l, data2 = read_txt(read_label2)
            y7 = data2
            # print("y7 shape " + str(y7.shape))
        else:
            y7 = np.ones(leng7) * 7

###########################################################################
    #splitting the training data and the validation data and printing it to check if it has the right shape
    # For combined data of all bead sizes
    if date == 0:  # For combined data
        s45 = int(np.round(len(y45) * split_ratio))
        s5 = int(np.round(len(y5) * split_ratio))
        s6 = int(np.round(len(y6) * split_ratio))
        s7 = int(np.round(len(y7) * split_ratio))
        if split_ratio == 0:
            x45 = x45[s45:, :]
            y45 = y45[s45:, :]
            x5 = x5[s5:, :]
            y5 = y5[s5:]
            x6 = x6[s6:, :]
            y6 = y6[s6:]
            x7 = x7[s7:, :]
            y7 = y7[s7:]
        else:
            x45_eval = x45[0:s45, :]
            y45_eval = y45[0:s45]
            x45 = x45[s45:, :]
            y45 = y45[s45:]
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

        x45 = np.transpose(x45)
        x5 = np.transpose(x5)
        x6 = np.transpose(x6)
        x7 = np.transpose(x7)

        #printing out the shape of the final data
        print("x45 shape train data (already split)" + str(x45.shape))
        print("x5 shape train data (already split)" + str(x5.shape))
        print("x6 shape train data (already split)" + str(x6.shape))
        print("x7 shape train data (already split)" + str(x7.shape))

        # combine 5,6,7um data in one array
        x45_eval = np.transpose(x45_eval)
        x5_eval = np.transpose(x5_eval)
        x6_eval = np.transpose(x6_eval)
        x7_eval = np.transpose(x7_eval)

    # For 4.5 and 6um beads
    if date == 1106:
        s45 = int(np.round(len(y45) * split_ratio))
        s6 = int(np.round(len(y6) * split_ratio))
        if split_ratio == 0:
            x45 = x45[s45:, :]
            y45 = y45[s45:, :]
            x6 = x6[s6:, :]
            y6 = y6[s6:]
        else:
            x45_eval = x45[0:s45, :]
            y45_eval = y45[0:s45]
            x45 = x45[s45:, :]
            y45 = y45[s45:]
            x6_eval = x6[0:s6, :]
            y6_eval = y6[0:s6]
            x6 = x6[s6:, :]
            y6 = y6[s6:]

        x45 = np.transpose(x45)
        x6 = np.transpose(x6)

        # printing out the shape of the final data
        print("x45 train data (already split) shape" + str(x45.shape))
        print("x6 train data (already split) shape" + str(x6.shape))

        # combine 5,6,7um data in one array
        x45_eval = np.transpose(x45_eval)
        x6_eval = np.transpose(x6_eval)

    # For 5,6,7 um beads
    if date == 1117 or date == 1207:
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

        x5 = np.transpose(x5)
        x6 = np.transpose(x6)
        x7 = np.transpose(x7)

        # printing out the shape of the final data
        print("x5 shape train data (already split)" + str(x5.shape))
        print("x6 shape train data (already split)" + str(x6.shape))
        print("x7 shape train data (already split)" + str(x7.shape))

        # combine 5,6,7um data in one array
        x5_eval = np.transpose(x5_eval)
        x6_eval = np.transpose(x6_eval)
        x7_eval = np.transpose(x7_eval)

############################################################################
    # Here below you define with what sets you want to train your network
    # e.g. If you don't include 6um so that you can test interpolation with 6um

    # All training data
    if exclude_size == 0:
        if date == 0:
            x_train = np.concatenate((x45,x5, x6, x7), axis=1)
            y_train = np.concatenate((y45,y5, y6, y7), axis=0)
            x_eval = np.concatenate((x45_eval,x5_eval, x6_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y45_eval,y5_eval, y6_eval, y7_eval), axis=0)
        elif date == 1106:
            x_train = np.concatenate((x45, x6), axis=1)
            y_train = np.concatenate((y45, y6), axis=0)
            x_eval = np.concatenate((x45_eval, x6_eval), axis=1)
            y_eval = np.concatenate((y45_eval, y6_eval), axis=0)
        elif date == 1117 or date == 1207:
            x_train = np.concatenate((x5, x6, x7), axis=1)
            y_train = np.concatenate((y5, y6, y7), axis=0)
            x_eval = np.concatenate((x5_eval, x6_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y5_eval, y6_eval, y7_eval), axis=0)

    ## exclude 4.5 um from training and evaluation set
    elif exclude_size == 45:
        if date == 0:
            x_train = np.concatenate((x5, x6, x7), axis=1)
            y_train = np.concatenate((y5, y6, y7), axis=0)
            x_eval = np.concatenate((x5_eval, x6_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y5_eval, y6_eval, y7_eval), axis=0)
        elif date == 1106:
            x_train = np.concatenate((x6), axis=1)
            y_train = np.concatenate((y6), axis=0)
            x_eval = np.concatenate((x6_eval), axis=1)
            y_eval = np.concatenate((y6_eval), axis=0)
        elif date == 1117 or date == 1207:
            x_train = np.concatenate((x5, x6, x7), axis=1)
            y_train = np.concatenate((y5, y6, y7), axis=0)
            x_eval = np.concatenate((x5_eval, x6_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y5_eval, y6_eval, y7_eval), axis=0)

    ## Exclude 5 um from training and evaluation set
    elif exclude_size == 5:
        if date == 0:
            x_train = np.concatenate((x45,  x6, x7), axis=1)
            y_train = np.concatenate((y45,  y6, y7), axis=0)
            x_eval = np.concatenate((x45_eval,  x6_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y45_eval,  y6_eval, y7_eval), axis=0)
        elif date == 1106:
            x_train = np.concatenate((x45, x6), axis=1)
            y_train = np.concatenate((y45, y6), axis=0)
            x_eval = np.concatenate((x45_eval, x6_eval), axis=1)
            y_eval = np.concatenate((y45_eval, y6_eval), axis=0)
        elif date == 1117 or date == 1207:
            x_train = np.concatenate(( x6, x7), axis=1)
            y_train = np.concatenate(( y6, y7), axis=0)
            x_eval = np.concatenate(( x6_eval, x7_eval), axis=1)
            y_eval = np.concatenate(( y6_eval, y7_eval), axis=0)

    ## Exclude 6um from training and evaluation set
    elif exclude_size == 6:
        if date == 0:
            x_train = np.concatenate((x45,x5, x7), axis=1)
            y_train = np.concatenate((y45,y5, y7), axis=0)
            x_eval = np.concatenate((x45_eval,x5_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y45_eval,y5_eval, y7_eval), axis=0)
        elif date == 1106:
            x_train = np.concatenate((x45 ), axis=1)
            y_train = np.concatenate((y45 ), axis=0)
            x_eval = np.concatenate((x45_eval ), axis=1)
            y_eval = np.concatenate((y45_eval), axis=0)
        elif date == 1117 or date == 1207:
            x_train = np.concatenate((x5, x7), axis=1)
            y_train = np.concatenate((y5, y7), axis=0)
            x_eval = np.concatenate((x5_eval, x7_eval), axis=1)
            y_eval = np.concatenate((y5_eval, y7_eval), axis=0)

    ## Exclude 7um from training and evaluation set
    elif exclude_size == 7:
        if date == 0:
            x_train = np.concatenate((x45,x5, x6), axis=1)
            y_train = np.concatenate((y45,y5, y6), axis=0)
            x_eval = np.concatenate((x45_eval,x5_eval, x6_eval), axis=1)
            y_eval = np.concatenate((y45_eval,y5_eval, y6_eval), axis=0)
        elif date == 1106:
            x_train = np.concatenate((x45, x6), axis=1)
            y_train = np.concatenate((y45, y6), axis=0)
            x_eval = np.concatenate((x45_eval, x6_eval), axis=1)
            y_eval = np.concatenate((y45_eval, y6_eval), axis=0)
        elif date == 1117 or date == 1207:
            x_train = np.concatenate((x5, x6), axis=1)
            y_train = np.concatenate((y5, y6), axis=0)
            x_eval = np.concatenate((x5_eval, x6_eval), axis=1)
            y_eval = np.concatenate((y5_eval, y6_eval), axis=0)

############################################################################
    #Finalising the trainingset
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


#####################################################################################
##################### Preparing test data ###########################################
#####################################################################################
def prep_test_data_allfreq(wdw, date_test, test_size):  # prepares the mixed beads data by putting frequencies next to each other

    if date_test == 1106:
        data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
    elif date_test == 1117:
        data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
    elif date_test == 1207:
        data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
    elif date_test == 0:
        data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"

    if test_size == 45:
        read_label = data_folder_test + "x_data_45_1.txt"
    elif test_size == 5:
        read_label = data_folder_test + "x_data_5.txt"
    elif test_size == 6:
        read_label = data_folder_test + "x_data_6.txt"
    elif test_size ==6_1:
        read_label = data_folder_test + "x_data_6_1.txt"
    elif test_size == 6_2:
        read_label = data_folder_test + "x_data_6_2.txt"
    elif test_size == 7:
        read_label = data_folder_test + "x_data_7.txt"
    elif test_size ==0:
        read_label = data_folder_test + "x1_data_mix.txt"

    l, data = read_txt(read_label)

    if date_test == 1106 and test_size == 6:
        x_mix = data
    else:
        x_mix = np.transpose(data)


    SIZE = wdw * 2  # Define the window size around the peaks as chosen in matlab
    leng = int(len(x_mix[:, 1]) / 6)
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
    # print("x_mix shape" + str(x.shape))

    return x


####################Test data
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
# data_folder_test = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"

### Calling the test data
x_test=prep_test_data_allfreq(data_folder_test)
# x_test=prep_test_data(data_folder_test)
print("test data shape" + str(x_test.shape))
save_test_file="C:\\Users\\rtene\\PycharmProjects\\Neural_network\\test_data_all_freq_6000.npy"
np.save(save_test_file,x_test)

################## Train data
x_train, y_train, x_eval, y_eval=make_training_set_allfreq_os(0.15)


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
#####################################################################################
#####################################################################################
#####################################################################################
###################################   Some old code below ###########################
#####################################################################################
#####################################################################################

# If using this data transpose everything and set window to 500 indicated as old data
# data_folder = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\"
# data_folder_test = "C:\\Users\\rtene\\PycharmProjects\\Neural_network\\"

####################Train / evaluation  data
# # data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
# # data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
# # data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"


# # x_train=np.concatenate((x_train0,x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7),axis=0)
# def make_training_set(data_folder, split_ratio):  # creates the data set required for training and evaluation
#     read_label = data_folder + "x_data_5.txt"  # the file that contains the 5um data
#     l, data = read_txt(read_label)
#     x5 = np.transpose(data)
#     y5 = np.ones((l[1])) * 5  # the labels for 5um
#
#     read_label = data_folder + "x_data_6.txt"  # the file that contains the 6um data
#     l, data = read_txt(read_label)
#     x6 = np.transpose(data)
#     y6 = np.ones((l[1])) * 6  # the labels for 6um
#
#     read_label = data_folder + "x_data_7.txt"  # the file that contains the 7um data
#     l, data = read_txt(read_label)
#     x7 = np.transpose(data)
#     y7 = np.ones((l[1])) * 7  # the labels for 7um
#
#     #combine all data as prep for training
#     x_train = np.concatenate((x5, x6, x7), axis=0)
#     y_train = np.concatenate((y5, y6, y7), axis=0)
#
#     # this block mixes the data
#     a = x_train
#     b = y_train
#     c = list(zip(a, b))
#     random.shuffle(c)
#     a, b = zip(*c)
#     x_train = np.array(a)
#     y_train = np.array(b)
#
#     # this splits the dataset to evaluation and training data set. split_ratio-> ration of evaluation to train data. Number between 0 and 1
#     s = int(np.round(len(y_train) * split_ratio))
#
#     if split_ratio == 0:
#         x_train = x_train[s:, :]
#         y_train = y_train[s:]
#         return x_train, y_train  # x=data y=labels
#     else:
#         x_eval = x_train[0:s, :]
#         y_eval = y_train[0:s]
#         x_train = x_train[s:, :]
#         y_train = y_train[s:]
#         return x_train, y_train, x_eval, y_eval

# ######################################################
#     # Making sure all data sets are equally large. If 2000 data points are trained on 6 mu, the 5.6 will be counted as a 6 sooner then when it is trained on equal footing
#     multiplier = 1
#
#     ids5 = np.arange(len(y5))
#     choices = np.random.choice(ids5, multiplier*min(leng5,leng6,leng7))
#     xx5 = x5[choices]
#     x5 = xx5
#
#     ids6 = np.arange(len(y6))
#     choices = np.random.choice(ids6, multiplier*min(leng5,leng6,leng7))
#     xx6 = x6[choices]
#     x6 = xx6
#
#     ids7 = np.arange(len(y7))
#     choices = np.random.choice(ids7, multiplier*min(leng5,leng6,leng7))
#     xx7 = x7[choices]
#     x7 = xx7
################################################################################


# def prep_test_data(data_folder_test):  # puts mixed data in one array (single frequency)
#     read_label = data_folder_test + "x1_data_mix.txt"
#     l, data = read_txt(read_label)
#     x = np.transpose(data)
#
#     return x

