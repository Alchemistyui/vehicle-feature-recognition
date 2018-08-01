import os
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io, data, color, exposure
import cv2
import h5py
import scipy
import pylab
from scipy.ndimage import filters
import scipy.signal as signal
import pylab as pl
# import math
# import pandas as pd
# from sklearn import preprocessing

imgs = []
imgs_origin = []
path = "/Users/ryshen/Desktop/车辆" #文件夹目录

def get_files(file_dir):
    cars = []
    
    for file in os.listdir(file_dir+'/车辆'): 
            cars.append(file_dir +'/车辆'+'/'+ file) 
    return  cars



def load_picture(path):
    global imgs
    global imgs_origin
    
    files= os.listdir(path) 
    
    # print(files)
    for file in files: #遍历文件夹
        # print(path+file)
        # print(os.path.isdir(path+file))
        if os.path.isdir(path+'/'+file): #判断是否是文件夹，不是文件夹才打开
            load_picture(path+'/'+file)

        else :
            img = cv2.imread(path+'/'+file, 0)
            img_origin = cv2.imread(path+'/'+file)
                # dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img)
            imgs_origin.append(img_origin)
   

def hann (cv2imgs):
    img_hann = []
    for i in range(len(cv2imgs)):
        # print(cv2imgs[i])
        img1 = cv2imgs[i].astype('float')
        img_h, img_w = img1.shape      
        hann =  signal.hann(img_w)
        hann2 = signal.hann(img_h)
        img2 = (img1.T * hann2).T * hann
        img_hann.append(img2)
    return img_hann


def dct_idct(img_hann):
    imgs = []
    for i in range(len(img_hann)):       
        img_dct = cv2.dct(img_hann[i])  
        sign = np.where(np.absolute(img_dct)<0, 0, img_dct)
        img_recor = cv2.idct(sign)   #进行离散余弦反变换
        imgs.append(img_recor)
    return imgs

# def idct(img_dct):
#     imgs = [];
#     for i in range(len(img_dct)):
#         img_recor = cv2.idct(img_dct[i])
#         imgs.append(img_recor) 
#     # 测试
#     # test = imgs[2].astype()
#     # io.imshow(imgs[2])
#     # io.show()

#     # plt.imshow(imgs[2],'gray')
#     # plt.title('IDCT2(cv2_idct)')
#     # plt.show()
#     # cv2.imshow('image2',imgs[2])
#     # cv2.waitKey(0)
#     # for i in range(len(img_dct)):
#     #     cv2.imshow('image',imgs[i])
#     #     cv2.waitKey(10)
#     # cv2.destroyAllWindows()
#     return imgs

def gauss(img_idct):
    imgs = []
    for i in range(len(img_idct)):
        square = np.square(img_idct[i]) 
        gauss = filters.gaussian_filter(square,0.005)
        normalizedImg = np.zeros((720, 1280))
        normalizedImg = cv2.normalize(gauss,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        imgs.append(normalizedImg)
    return imgs

def dilation(img_gauss):
    imgs = []
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_dilation = np.ones((15, 15), np.uint8)
    for i in range(len(img_gauss)):
        ret,thresh=cv2.threshold(img_gauss[i],50,255,cv2.THRESH_BINARY) 
        img_open = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel_open)
        img_dilation = cv2.dilate(img_open,kernel_dilation,iterations = 1)
        imgs.append(img_dilation)
    return imgs

def fin_counter(img_dilation, cv2imgs_origin, train_dir):
    for i in range(len(img_dilation)): 
        bin8bit = img_dilation[i].astype(np.uint8)
        ret, contours, hierarchy = cv2.findContours(bin8bit,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_size = 0
        max_rect = [0, 0, 0, 0]
        for j in range(len(contours)):
            #获得正外接矩形的左上角坐标及宽高  
            x, y, w, h = cv2.boundingRect(contours[j])
            if w*h > max_size and w*h:
                max_rect[0] = x 
                max_rect[1] = y
                max_rect[2] = w
                max_rect[3] = h
                max_size = w*h
        # 用画矩形方法绘制正外接矩形
        # print(max_rect)
        # print(cv2imgs_origin[i])
        # io.imshow(cv2imgs_origin[i])
        # # # io.imshow(green)
        # io.show() 
        green = cv2.rectangle(cv2imgs_origin[i], (int(max_rect[0]*0.9), int(max_rect[1]*0.9)), (max_rect[0]+int(max_rect[2]*1.2), max_rect[1]+int(max_rect[3]*1.2)), (0, 255, 0), 3);

        cutImg = cv2imgs_origin[i][max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]
        cv2.imwrite(train_dir+'/out/'+str(i)+'.png', cutImg)

def td(img, dir_x, dir_y):
    h, w = img.shape
    sum = 0
    for x in range(w-2):
        for y in range(h-2):
            sum = sum + np.square(img[y,x] - img[y-dir_y, x-dir_x] - img[y+dir_y, x+dir_x])
    return sum/(h*w)

def sobel_fun(img_gauss):
    imgs = []
    for i in range(len(img_gauss)):

        sobel_x = cv2.Sobel(img_gauss[i],cv2.CV_16S,1,0)
        sobel_y = cv2.Sobel(img_gauss[i],cv2.CV_16S,0,1)

        # sobel_x = cv2.Sobel(img,cv2.CV_16S,1,0)
        # sobel_y = cv2.Sobel(sobel_x,cv2.CV_16S,0,1)

        # return cv2.convertScaleAbs(sobel_y)

        # print(td(sobel_y, 0, 1)-td(sobel_x, 1, 0))
        
        if td(sobel_y, 0, 1)-td(sobel_x, 1, 0) > 150:
            print('sobel_x')
            # return cv2.convertScaleAbs(sobel_y)
            # io.imshow(cv2.convertScaleAbs(sobel_x))
            # io.show()
            # io.imshow(cv2.convertScaleAbs(sobel_y))
            # io.show()
            # io.imshow(cv2.convertScaleAbs(sobel_y))
            # io.show()
            img = cv2.convertScaleAbs(sobel_x)
             
        elif td(sobel_y, 0, 1)-td(sobel_x, 1, 0) > 50:
            print('sobel_y')
            # return cv2.convertScaleAbs(sobel_x)   # 转回uint8
            # io.imshow(cv2.convertScaleAbs(sobel_y))
            # io.show() 
            sobel_xx = cv2.Sobel(sobel_y,cv2.CV_16S,1,0)
            img = cv2.convertScaleAbs(sobel_xx)
            # return img
            
        else :
            print('else')
            # return cv2.convertScaleAbs(sobel_x)
            img = cv2.convertScaleAbs(sobel_y)
        imgs.append(img)
    return imgs






# train_dir = '/Users/ryshen/Desktop'
# image_list = get_files(train_dir)


# cv2imgs, cv2imgs_origin = load_picture(path)
load_picture(path)
img_hann = hann(imgs)
# img_idct = dct_idct(img_hann)
img_gauss = gauss(img_hann)
img_sobel = sobel_fun(img_gauss)
img_dilation = dilation(img_sobel)
fin_counter(img_dilation, imgs_origin, train_dir)




