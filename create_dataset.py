import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io, data, color
import cv2
import h5py
import scipy
import pylab
from scipy.ndimage import filters

def get_files(file_dir):
    cars = []
    
    for file in os.listdir(file_dir+'/数据'): 
            cars.append(file_dir +'/数据'+'/'+ file) 
    return  cars


def create_h5(file_dir):
    img_rows = 1280
    img_cols = 720
    # image =  np.random.rand(len(image_list), 720, 1280, 3).astype('float32')
    # for i in range(len(image_list)):
    #     image[i] = np.array(plt.imread(image_list[i]))
    testLen = 2
    Train_image =  np.random.rand(len(image_list)-testLen, 720, 1280, 3).astype('float32')
    Test_image =  np.random.rand(testLen, 720, 1280, 3).astype('float32')
    for i in range(len(image_list)-testLen):
        Train_image[i] = np.array(plt.imread(image_list[i]))
     
    for i in range(len(image_list)-testLen, len(image_list)):
        Test_image[i+testLen-len(image_list)] = np.array(plt.imread(image_list[i]))

    # Create a new file
    f = h5py.File(file_dir + 'data.h5', 'w')
    # f.create_dataset('cars', data=image)
    f.create_dataset('X_train', data=Train_image)
    f.create_dataset('X_test', data=Test_image)
    f.close()

def load_h5(file_dir):
    dataset = h5py.File(file_dir + 'data.h5', 'r')
    # set_x_orig = np.array(dataset['cars'][:])

    train_set_x_orig = np.array(dataset['X_train'][:]) # your train set features
    test_set_x_orig = np.array(dataset['X_test'][:]) # your train set features
    dataset.close()
    print(train_set_x_orig.shape)
    print(test_set_x_orig.shape)
    #测试
    # plt.imshow(train_set_x_orig[2])
    # pylab.show()

def load_picture(image_list):
    imgs = [];
    for i in range(len(image_list)):
        img = cv2.imread(image_list[i])
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(dst)
    # 测试
    # for i in range(len(imgs)):
    #     cv2.imshow('image',imgs[i])
    #     cv2.waitKey(5)
    #     cv2.destroyAllWindows()
    return imgs


def dct(cv2imgs):
    imgs = []
    imgs_log = []
    for i in range(len(cv2imgs)):
        img = cv2imgs[i].astype('float')
        img_dct = cv2.dct(img)    
        img_dct_log = np.log(abs(img_dct))  #进行log处理
        imgs.append(img_dct_log)
        imgs_log.append(img_dct)
    # 测试
    # cv2.imshow('image2',cv2imgs[2].astype('float'))
    # cv2.waitKey(0)
    # for i in range(len(cv2imgs)):
    #     cv2.imshow('image',imgs[i])
    #     cv2.waitKey(10)
    cv2.destroyAllWindows()
    return imgs,imgs_log

def idct(img_dct):
    imgs = [];
    for i in range(len(img_dct)):
        img_recor = cv2.idct(img_dct[i])
        imgs.append(img_recor) 
    # 测试
    # cv2.imshow('image2',imgs[2])
    # cv2.waitKey(0)
    # for i in range(len(img_dct)):
    #     cv2.imshow('image',imgs[i])
    #     cv2.waitKey(10)
    #     cv2.destroyAllWindows()
    return imgs

def sign(img_dct_log):
    imgs = []
    for i in range(len(img_dct_log)):
        sign = np.sign(img_dct_log[i])
        imgs.append(sign)
    return imgs



def test(cv2imgs):
    img = cv2.imread(image_list[2])
    img1 = img.astype('float')
    dst = color.rgb2gary(img)
    print(dst.dtype)
    io.imshow(dst)
    io.show()  

    img_dct = cv2.dct(img1)         #进行离散余弦变换
     
    img_dct_log = np.log(abs(img_dct))  #进行log处理
     
    img_recor = cv2.idct(img_dct)    #进行离散余弦反变换

    cv2.imshow('image3',img)
    cv2.waitKey(0)
    cv2.imshow('image2',img1)
    cv2.waitKey(0)
    # cv2.imshow('image',img_recor)
    # cv2.waitKey(0)
    # cv2.imshow('image1',img_dct)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


#两个回调函数
def GaussianBlur(sign_img):
    gauss = []
    for i in range(len(sign_img)):
        square = np.square(sign_img[i])
        img = filters.gaussian_filter(square,0.05)
        gauss.append(img)
    return gauss

def Normalize(data):
    imgs = []
    for i in range(len(data)):
        m = np.mean(data[i])
        mx = max(data[i])
        mn = min(data[i])
        imgs.append((float(j) - m) / (mx - mn) * 255 for j in data)


# def filter(normal):
    # sum = 0
    # for i in range(len(normal)):   
    #     for j in range(len(normal[i]):
    #         sum = sum+normal[i][j]
    #     avg = sum/b
    #     for j in range(len(normal[i]):
    #         if (normal[i] < (avg/2))




train_dir = '/Users/ryshen/Desktop'
image_list = get_files(train_dir)
 
# print(image_list)
# print(len(image_list))
create_h5(train_dir)
load_h5(train_dir)
cv2imgs = load_picture(image_list)
img_dct, img_dct_log = dct(cv2imgs)
sign_img = sign(img_dct_log)
gauss_img = GaussianBlur(sign_img)
# normal = Normalize(gauss_img)
# idct_img = idct(sign_img)
test(cv2imgs)
# print(len(label_list))





