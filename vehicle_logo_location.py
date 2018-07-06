
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

import h5py
import scipy

def get_files(file_dir):
    cars = []
    # label_cats = []
    # dogs = []
    # label_dogs = []
    # file_dir = "/Users/ryshen/Desktop"
    
    for file in os.listdir(file_dir+'/数据'):
            # cars.append(file_dir +'/数据'+'/'+ file) 
            cars.append(file_dir +'/数据'+'/'+ file) 
            # label_cats.append(0)     #添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    
            
    #把cat和dog合起来组成一个list（img和lab）
    # image_list = np.hstack((cats, dogs))
    # label_list = np.hstack((label_cats, label_dogs))
 
    #利用shuffle打乱顺序
    # temp = np.array([image_list, label_list])
    # temp = temp.transpose()
    # np.random.shuffle(temp)
 
    #从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list] 
    
    return  cars
    #返回两个list 分别为图片文件名及其标签  顺序已被打乱


train_dir = '/Users/ryshen/Desktop'
image_list = get_files(train_dir)
 
# print(image_list)
print(len(image_list))
# print(len(label_list))







# #450为数据长度的20%
# testLength = len(image_list)*0.8;
# Train_image =  np.random.rand(len(image_list) - testLength, 1280, 720, 3)
# # .astype('float32')
# # Train_label = np.random.rand(len(image_list)-450, 1).astype('float32')
 
# Test_image =  np.random.rand(len(testLength, 1280, 720, 3).astype('float32'))
# # Test_label = np.random.rand(450, 1).astype('float32')





# # for i in range(len(image_list)-450):
# #     Train_image[i] = np.array(plt.imread(image_list[i]))

# #     Train_label[i] = np.array(label_list[i])


# for i in range(len(image_list)-testLength):
#     Train_image[i] = np.array(plt.imread(image_list[i]))
#     # Train_label[i] = np.array(label_list[i])
 
# for i in range(len(image_list) - testLength, len(image_list)):
#     Test_image[i+testLength-len(image_list)] = np.array(plt.imread(image_list[i]))
#     # Test_label[i+450-len(image_list)] = np.array(label_list[i])

img_rows = 1280
img_cols = 720
# testLength = len(image_list)*0.8;
image =  np.random.rand(len(image_list), 720, 1280, 3).astype('float32')
for i in range(len(image_list)):
    image[i] = np.array(plt.imread(image_list[i]))
# np.random.rand(len(image_list) - testLength, 1280, 720, 3)
# Create a new file
f = h5py.File(train_dir + 'data.h5', 'w')
# f.create_dataset('X_train', data=Train_image)
# f.create_dataset('y_train', data=Train_label)
f.create_dataset('cars', data=image)
# f.create_dataset('y_test', data=Test_label)
f.close()

# Load hdf5 dataset
train_dataset = h5py.File(train_dir + 'data.h5', 'r')
set_x_orig = np.array(train_dataset['cars'][:]) # your train set features
# train_set_y_orig = np.array(train_dataset['y_train'][:]) # your train set labels
# test_set_x_orig = np.array(train_dataset['X_test'][:]) # your train set features
# test_set_y_orig = np.array(train_dataset['y_test'][:]) # your train set labels
f.close()



print(set_x_orig.shape)
# print(train_set_y_orig.shape)
 
# print(train_set_x_orig.max())
# print(train_set_x_orig.min())
 
# print(test_set_x_orig.shape)
# print(test_set_y_orig.shape)

#测试
plt.imshow(set_x_orig[2])
# print(train_set_y_orig[222])






