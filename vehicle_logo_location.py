
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

import h5py
import scipy

def get_files(file_dir):
    cars = []
    
    for file in os.listdir(file_dir+'/数据'): 
            cars.append(file_dir +'/数据'+'/'+ file) 
    return  cars


def create_h5(file_dir):
    img_rows = 1280
    img_cols = 720
    image =  np.random.rand(len(image_list), 720, 1280, 3).astype('float32')
    for i in range(len(image_list)):
        image[i] = np.array(plt.imread(image_list[i]))
    # Create a new file
    f = h5py.File(file_dir + 'data.h5', 'w')
    f.create_dataset('cars', data=image)
    f.close()

def load_h5(file_dir):
    dataset = h5py.File(train_dir + 'data.h5', 'r')
    set_x_orig = np.array(dataset['cars'][:])
    dataset.close()
    print(set_x_orig.shape)
    #测试
    plt.imshow(set_x_orig[2])




train_dir = '/Users/ryshen/Desktop'
image_list = get_files(train_dir)
 
# print(image_list)
print(len(image_list))
create_h5(train_dir)
load_h5(train_dir)
# print(len(label_list))



