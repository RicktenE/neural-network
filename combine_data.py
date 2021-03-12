import numpy as np

def read_txt(filepath):  # reads a txt file line by line and returns the data in an array and the length of the array
    a = np.loadtxt(filepath)
    data = a  # choosing the second column
    l = np.shape(data)
    return l, data


###########Creating combined data files from 11-06
########### Here 4,5 1 and 4,5 2 from 11-06 are combined into x_data_45.txt
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
read_label = data_folder + "x_data_45_1.txt"
l, data = read_txt(read_label)
x_45_1 = np.transpose(data)
# x_45_1 = np.array(data)

read_label = data_folder + "x_data_45_2.txt"
l, data = read_txt(read_label)
x_45_2 = np.transpose(data)
# x_45_2 = np.array(data)

x_45 = np.concatenate((x_45_1, x_45_2), axis=0)

print('shape x_45_1', str(x_45_1.shape))
print('shape x_45_2', str(x_45_2.shape))


x_45 = np.transpose(x_45)# to keep all data in the same shape as the non combined data
print('shape x_45 combined 11-06', str(x_45.shape))
np.savetxt(data_folder + 'x_data_45.txt', x_45, delimiter='\t', newline='\n')

##########Here 6 1 and 6 2 from 11-06 are combined into x_data_6.txt
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

x_6 = np.transpose(x_6)# to keep all data in the same shape as the non combined data
print('shape x_6 combined 11-06', str(x_6.shape))
np.savetxt( data_folder + 'x_data_6 .txt', x_6, delimiter='\t', newline='\n')

########### Here mixed 1 and mixed 2 from 11-06 are combined into x_data_mix.txt
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
read_label = data_folder + "x1_data_mix_1.txt"
l, data = read_txt(read_label)
x_mix_1 = np.transpose(data)
# x_mix_1 = np.array(data)

read_label = data_folder + "x1_data_mix_2.txt"
l, data = read_txt(read_label)
x_mix_2 = np.transpose(data)
# x_mix_2 = np.array(data)

x_mix = np.concatenate((x_mix_1, x_mix_2), axis=0)

print('shape x_mix_1', str(x_mix_1.shape))
print('shape x_mix_2', str(x_mix_2.shape))
x_mix = np.transpose(x_mix)# to keep all data in the same shape as the non combined data

print('shape x_mix 11-06 (4,5um and 6um)', str(x_mix.shape))
np.savetxt(data_folder + 'x1_data_mix.txt', x_mix, delimiter='\t', newline='\n')

###########Creating combined data files from 12-07
########### Here mmixed 1 and 2 from 12-07 are combined into x1_data_mix.txt
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
read_label = data_folder + "x1_data_mix_1.txt"
l, data = read_txt(read_label)
x_mix_1 = np.transpose(data)
# x_mix_1 = np.array(data)

read_label = data_folder + "x1_data_mix_2.txt"
l, data = read_txt(read_label)
x_mix_2 = np.transpose(data)
# x_mix_2 = np.array(data)

x_mix = np.concatenate((x_mix_1, x_mix_2), axis=0)

# printing the shapes to check if everything goes well
print('shape x_mix_1', str(x_mix_1.shape))
print('shape x_mix_2', str(x_mix_2.shape))

x_mix = np.transpose(x_mix)# to keep all data in the same shape as the non combined data
print('shape x_mix 12-07 (5um,6um,7um)', str(x_mix.shape))
np.savetxt(data_folder + 'x1_data_mix.txt', x_mix, delimiter='\t', newline='\n')



###########Creating the combined data files where all dates are included
###########Here the x_data_5.txt files from all dates are combined into x_data_5.txt
###########11-06 has no 5 um data

# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# read_label = data_folder + "x_data_5.txt"
# l, data = read_txt(read_label)
# x_5_1 = np.transpose(data)
# # x_5_1 = np.array(data)

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
read_label = data_folder + "x_data_5.txt"
l, data = read_txt(read_label)
x_5_2 = np.transpose(data)
# x_5_2 = np.array(data)

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
read_label = data_folder + "x_data_5.txt"
l, data = read_txt(read_label)
x_5_3 = np.transpose(data)
# x_5_3 = np.array(data)
# x_5 = np.concatenate((x_5_1, x_5_2, x_5_3), axis=0)
x_5 = np.concatenate((x_5_2, x_5_3), axis=0)

# printing the shapes to check if everything goes well
# print('shape x_5_1', str(x_5_1.shape))
print('shape x_5_2', str(x_5_2.shape))
print('shape x_5_3', str(x_5_3.shape))

x_5 = np.transpose(x_5)# to keep all data in the same shape as the non combined data
print('shape x_5 combined all dates', str(x_5.shape))
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"
np.savetxt(data_folder + 'x_data_5.txt', x_5, delimiter='\t', newline='\n')

########### Here the x_data_6.txt files from all dates are combined into x_data_6.txt
########### All 3 dates have 6um beads
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
read_label = data_folder + "x_data_6.txt"
l, data = read_txt(read_label)
# x_6_1 = np.transpose(data)
x_6_1 = np.array(data)
# print('shape x_6_1 combined 11-06 read out', str(x_6_1.shape))

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
read_label = data_folder + "x_data_6.txt"
l, data = read_txt(read_label)
x_6_2 = np.transpose(data)
# x_6_2 = np.array(data)

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
read_label = data_folder + "x_data_6.txt"
l, data = read_txt(read_label)
x_6_3 = np.transpose(data)
# x_6_3 = np.array(data)
x_6 = np.concatenate((x_6_1, x_6_2, x_6_3), axis=0)


# printing the shapes to check if everything goes well
print('shape x_6_1', str(x_6_1.shape))
print('shape x_6_2', str(x_6_2.shape))
print('shape x_6_3', str(x_6_3.shape))


x_6 = np.transpose(x_6)# to keep all data in the same shape as the non combined data
print('shape x_6 combined all dates', str(x_6.shape))

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"
np.savetxt(data_folder + 'x_data_6.txt', x_6, delimiter='\t', newline='\n')

########### Here the x_data_7.txt files from all dates are combined into x_data_7.txt
# ##########11-06 has no 7 um data
# data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
# read_label = data_folder + "x_data_5.txt"
# l, data = read_txt(read_label)
# x_5_1 = np.transpose(data)
# # x_5_1 = np.array(data)

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
read_label = data_folder + "x_data_7.txt"
l, data = read_txt(read_label)
x_7_2 = np.transpose(data)
# x_7_2 = np.array(data)

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
read_label = data_folder + "x_data_7.txt"
l, data = read_txt(read_label)
x_7_3 = np.transpose(data)
# x_7_3 = np.array(data)
# x_7 = np.concatenate((x_7_1, x_7_2, x_7_3), axis=0)
x_7 = np.concatenate((x_7_2, x_7_3), axis=0)

# printing the shapes to check if everything goes well
# print('shape x_7_1', str(x_7_1.shape))
print('shape x_7_2', str(x_7_2.shape))
print('shape x_7_3', str(x_7_3.shape))


x_7 = np.transpose(x_7) # to keep all data in the same shape as the non combined data
print('shape x_7 combined all dates', str(x_7.shape))
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"
np.savetxt(data_folder + 'x_data_7.txt', x_7, delimiter='\t', newline='\n')

########### Here the x1_data_mix.txt files from all dates are combined into x1_data_mix.txt
########### All 3 dates have mixed um beads
########### But 11-06 has mixed 4.5um and 6um
########## And 11-17 and 12-07 have 5,6,7um mixed. Therefore I leave out 11-06
## data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201106\\"
## read_label = data_folder + "x_data_6.txt"
## l, data = read_txt(read_label)
# # x_6_1 = np.transpose(data)
###x_6_1 = np.array(data) # this is already transposed from the previous combining of the data of 6um of 11-06

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201117\\"
read_label = data_folder + "x1_data_mix.txt"
l, data = read_txt(read_label)
x_mix_2 = np.transpose(data)
# x_mix_2 = np.array(data)

data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\20201207\\"
read_label = data_folder + "x1_data_mix.txt"
l, data = read_txt(read_label)
x_mix_3 = np.transpose(data)
# x_mix_3 = np.array(data)
# x_mix = np.concatenate((x_mix_1, x_mix_2, x_mix_3), axis=0)
x_mix = np.concatenate((x_mix_2, x_mix_3), axis=0)


# printing the shapes to check if everything goes well
# print('shape x_6_1', str(x_6_1.shape))
print('shape x_mix_2', str(x_mix_2.shape))
print('shape x_mix_3', str(x_mix_3.shape))
# print('shape x1_data_mix combined all dates', str(x_mix.shape))


x_mix = np.transpose(x_mix) # to keep all data in the same shape as the non combined data
print('shape x1_data_mix combined 11-17 and 12-07', str(x_mix.shape))
data_folder = "D:\\Saxion\\Jaar 4\\Bachelor Thesis\\Data Rick\\combined\\"
np.savetxt(data_folder + 'x1_data_mix.txt', x_mix, delimiter='\t', newline='\n')


