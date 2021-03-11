import numpy as np
import json

def read_txt(filepath):  # reads a txt file line by line and returns the data in an array and the length of the array
    a = np.loadtxt(filepath)
    data = a  # choosing the second column
    l = np.shape(data)
    return l, data


#Creating combined data files from 11-06
## Here 4,5 1 and 4,5 2 from 11-06 are combined into x_data_45.txt
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# read_label = data_folder + "x_data_45_1.txt"
# l, data = read_txt(read_label)
# x_45_1 = np.transpose(data)
# # x_45_1 = np.array(data)
#
# read_label = data_folder + "x_data_45_2.txt"
# l, data = read_txt(read_label)
# x_45_2 = np.transpose(data)
# # x_45_2 = np.array(data)
#
# x_45 = np.concatenate((x_45_1, x_45_2), axis=0)
#
# print('shape x_45_1', str(x_45_1.shape))
# print('shape x_45_2', str(x_45_2.shape))
# print('shape x_45', str(x_45.shape))

# np.savetxt(data_folder + 'x_data_45.txt', x_45, delimiter='\t', newline='\n')

## Here 6 1 and 6 2 from 11-06 are combined into x_data_6.txt
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
read_label = data_folder + "x_data_6_1.txt"
l, data = read_txt(read_label)
x_6_1 = np.transpose(data)
# x_6_1 = np.array(data)

read_label = data_folder + "x_data_6_2.txt"
l, data = read_txt(read_label)
x_6_2 = np.transpose(data)
# x_6_2 = np.array(data)

x_6 = np.concatenate((x_6_1, x_6_2), axis=0)

print('shape x_6_1', str(x_6_1.shape))
print('shape x_6_2', str(x_6_2.shape))
print('shape x_6', str(x_6.shape))

np.savetxt( data_folder + 'x_6 .txt', x_6, delimiter='\t', newline='\n')

## Here mixed 1 and mixed 2 from 11-06 are combined into x_data_mix.txt
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# read_label = data_folder + "x1_data_mix_1.txt"
# l, data = read_txt(read_label)
# x_mix_1 = np.transpose(data)
# # x_mix_1 = np.array(data)
#
# read_label = data_folder + "x1_data_mix_2.txt"
# l, data = read_txt(read_label)
# x_mix_2 = np.transpose(data)
# # x_mix_2 = np.array(data)
#
# x_mix = np.concatenate((x_mix_1, x_mix_2), axis=0)
#
# print('shape x_mix_1', str(x_mix_1.shape))
# print('shape x_mix_2', str(x_mix_2.shape))
# print('shape x_mix', str(x_mix.shape))

# np.savetxt(data_folder + 'x1_data_mix.txt', x_mix, delimiter='\t', newline='\n')

#Creating combined data files from 12-07
## Here mmixed 1 and 2 from 12-07 are combined into x1_data_mix.txt
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
# read_label = data_folder + "x1_data_mix_1.txt"
# l, data = read_txt(read_label)
# x_mix_1 = np.transpose(data)
# # x_mix_1 = np.array(data)
#
# read_label = data_folder + "x1_data_mix_2.txt"
# l, data = read_txt(read_label)
# x_mix_2 = np.transpose(data)
# # x_mix_2 = np.array(data)
#
# x_mix = np.concatenate((x_mix_1, x_mix_2), axis=0)
#
# print('shape x_mix_1', str(x_mix_1.shape))
# print('shape x_mix_2', str(x_mix_2.shape))
# print('shape x_mix', str(x_mix.shape))

# np.savetxt(data_folder + 'x1_data_mix.txt', x_mix, delimiter='\t', newline='\n')



#Creating the combined data files where all dates are included
## Here the x_data_5.txt files from all dates are combined into x_data_5.txt
# data_folder1 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# data_folder2 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
# data_folder3 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"

## Here the x_data_6.txt files from all dates are combined into x_data_6.txt
# data_folder1 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# data_folder2 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
# data_folder3 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"

## Here the x_data_7.txt files from all dates are combined into x_data_7.txt
# data_folder1 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# data_folder2 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
# data_folder3 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"

## Here the x1_data_mix.txt files from all dates are combined into x1_data_mix.txt

# data_folder1 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# data_folder2 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
# data_folder3 = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"


