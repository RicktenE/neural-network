import numpy as np

# import time
np.set_printoptions(precision=3, suppress=True)


############################################################################################################
############################################################################################################
####################################### configure the setup ################################################
############################################################################################################
############################################################################################################
# wdw_all = [200, 300, 400, 500, 600, 700, 800]
# wdw_all = [300,400]
# date_multiple = [1117,0]
# learning_rate_all =[0.1, 0.01, 0.001, 0.0001, 0.00001]
# nodes_all = [2,4,6,8,10,12,14,16]
nodes_all = [2,6,12,14]
# act_func_all = ['selu', 'elu', 'linear']
# len(act_func_all)
# split_ratio_all = [0.25, 0.20, 0.15, 0.10, 0.05, 0]
default = [1]
for q in range(len(default)):
    for k in range(len(default)):
        for i in range(len(nodes_all)):
            wdw = 300                  # window size as chosen in matlab

            #configuring the train data
            date = 1117                      # selecting the training data, 0 = combined
            corrected_values = False        # True for using the corrected diameters False for using exact diameters
            split_ratio = 0.15              # The percentage of validation data from the trianing set
            exclude_size = 7                # 0 to exclude nothing, 45 for 4.5, 5 for 5, 6 for 6, 7 for 7
            # exclude_day = 0       does not work yet # 0 to exclude nothing
            frequency_count = 6

            #configuring the test data
            date_test = 1117               # selecting the test data 0 for combined
            test_size = 0                  # selecting the test size 0= mixed

            #configuring the network
            act_func = 'linear'            # Activation function used in training and testing
            learning_rate = 0.001           # the learning rate of the training
            patience = 250                  # the amount of epochs the training has to continue after finding the lowest val_loss
            epochs = 10000                   # max number of epochs before the training stops
            nodes = nodes_all[i]

            #What to run:
            train_ = 1 # 1 if you want to train the network again
            test_ = 1 # 1 if you want to test

            ############################################################################################################
            ############################################################################################################
            ################################ calling the functions #####################################################
            ############################################################################################################
            ############################################################################################################

            from prep import make_training_set_allfreq_os
            from prep import prep_test_data_allfreq
            from training import train
            from testing import test
            from win32com.client import Dispatch

            #################################
            # import pyttsx3
            # engine = pyttsx3.init()
            # engine.say("Your program has finished")
            # engine.runAndWait()
            ####################################

            # speak = Dispatch("SAPI.SpVoice").Speak
            #
            # speak("Your program terminated")
            #################################
            # from gtts import gTTS
            # import os
            #
            # tts = gTTS(text="This is the pc speaking", lang='en')
            # tts.save("pcvoice.mp3")
            # os.system("start pcvoice.mp3")
            # #################################
            if  train_ == 1 and test_ == 1:
                make_training_set_allfreq_os(wdw,corrected_values,date,exclude_size,split_ratio,frequency_count)
                prep_test_data_allfreq(wdw,date_test,test_size,frequency_count)
                train(wdw,act_func,learning_rate,patience,epochs,nodes)
                test(wdw,act_func,date,exclude_size,date_test,test_size,nodes,frequency_count)

            elif train_ != 1 and test_ == 1:
                prep_test_data_allfreq(wdw,date_test,test_size,frequency_count)
                test(wdw,act_func,date,exclude_size,date_test,test_size,nodes,frequency_count)

            elif train_ == 1 and test_ != 1:
                make_training_set_allfreq_os(wdw,corrected_values,date,exclude_size,split_ratio,frequency_count)
                train(wdw,act_func,learning_rate,patience,epochs,nodes)

speak = Dispatch("SAPI.SpVoice").Speak

speak("Finished calculations")

