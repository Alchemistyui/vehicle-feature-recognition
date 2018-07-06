
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import h5py
import scipy

def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    
    for file in os.listdir(file_dir+'/not_tumble'):
            cats.append(file_dir +'/not_tumble'+'/'+ file) 
            label_cats.append(0)     #添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    for file in os.listdir(file_dir+'/yes_tumble'):
            dogs.append(file_dir +'/yes_tumble'+'/'+file)
            label_dogs.append(1)
            
    #把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
 
    #利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list] 
    
    return  image_list,label_list
    #返回两个list 分别为图片文件名及其标签  顺序已被打乱