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

# 存放灰度图片的数组
imgs = []
# 存放原始图片的数组
imgs_origin = []
#输入数据文件夹目录
path = "/Users/ryshen/Desktop/粗定位" 
#存放数据文件夹目录
train_dir = '/Users/ryshen/Desktop'

# def get_files(file_dir):
#     cars = []
    
#     for file in os.listdir(file_dir+'/车辆'): 
#             cars.append(file_dir +'/车辆'+'/'+ file) 
#     return  cars


# 将粗定位图片读入至两个数组
def load_picture(path):
    global imgs
    global imgs_origin
    
    files= os.listdir(path) 
    
    # print(files)
    for file in files: #遍历文件夹
        img = cv2.imread(path+'/'+file, 0)
        img_origin = cv2.imread(path+'/'+file)
        # dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
        imgs_origin.append(img_origin)


        # if os.path.isdir(path+'/'+file): 
        #     load_picture(path+'/'+file)

        # else :
        #     img = cv2.imread(path+'/'+file, 0)
        #     img_origin = cv2.imread(path+'/'+file)
        #         # dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     imgs.append(img)
        #     imgs_origin.append(img_origin)
   

# 对灰度图进行汉宁窗处理
def hann (cv2imgs):
    img_hann = []
    for i in range(len(cv2imgs)):
        # print(cv2imgs[i])
        img1 = cv2imgs[i].astype('float')
        img_h, img_w = img1.shape 
        # 为对中心进行汉宁窗需要两个hann数组
        hann =  signal.hann(img_w)
        hann2 = signal.hann(img_h)
        img2 = (img1.T * hann2).T * hann
        img_hann.append(img2)
    return img_hann

#进行离散余弦反变换
def dct_idct(img_hann):
    imgs = []
    for i in range(len(img_hann)):       
        img_dct = cv2.dct(img_hann[i])  
        sign = np.where(np.absolute(img_dct)<0, 0, img_dct)
        img_recor = cv2.idct(sign)   
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

# 进行高斯模糊
def gauss(img_idct):
    imgs = []
    for i in range(len(img_idct)):
        # 先对像素值进行平方以扩大差值
        square = np.square(img_idct[i]) 
        gauss = filters.gaussian_filter(square,0.005)
        normalizedImg = np.zeros((720, 1280))
        normalizedImg = cv2.normalize(gauss,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        imgs.append(normalizedImg)
    return imgs

# 开闭操作
def dilation(img_gauss):
    imgs = []
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_dilation = np.ones((15, 15), np.uint8)
    # kernel_open = np.ones((5, 5), np.uint8)
    # kernel_erosion = np.ones((9, 9), np.uint8)
    # kernel_dilation = np.ones((15, 15), np.uint8)
    # kernel = np.ones((3, 3), np.uint8)

    for i in range(len(img_gauss)):
        ret,thresh=cv2.threshold(img_gauss[i],50,255,cv2.THRESH_BINARY) 
        # img_open = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel_open)
        # img_erosion = cv2.erode(img_open,kernel_erosion,iterations = 1)
        # img_dilation = cv2.dilate(thresh,kernel_dilation,iterations = 1)
        # img_dilation = cv2.dilate(thresh,kernel,iterations = 1)
        # img_close = cv2.morphologyEx(gauss,cv2.MORPH_CLOSE,kernel_open)

        img_open = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel_open)
        img_dilation = cv2.dilate(img_open,kernel_dilation,iterations = 1)
        imgs.append(img_dilation)
    return imgs

# 获得最后的车标位置
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

                plate_contour_minRectangle = cv2.minAreaRect(contours[j])
                plate_points = cv2.boxPoints(plate_contour_minRectangle).astype(int)
                print(plate_points)   
        # 用画矩形方法绘制正外接矩形
        # print(max_rect)
        # print(cv2imgs_origin[i])
        # io.imshow(cv2imgs_origin[i])
        # # # io.imshow(green)
        # io.show() 
        # green = cv2.rectangle(cv2imgs_origin[i], (int(max_rect[0]*0.9), int(max_rect[1]*0.9)), (max_rect[0]+int(max_rect[2]*1.2), max_rect[1]+int(max_rect[3]*1.2)), (0, 255, 0), 3);
        # green = cv2.rectangle(cv2imgs_origin[i], (int(max_rect[0]), int(max_rect[1])), (max_rect[0]+int(max_rect[2]), max_rect[1]+int(max_rect[3])), (0, 255, 0), 3);




        cutImg = cv2imgs_origin[i][max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]
        # cutImg = cv2imgs_origin[i][int(max_rect[1]*0.9):max_rect[1]+int(max_rect[3]*1.2), int(max_rect[0]*0.9):max_rect[0]+int(max_rect[2])]
        cv2.imwrite(train_dir+'/out/'+str(i)+'.png', cutImg)

        


# sobel算子判断横竖条纹的辅助函数
def td(img, dir_x, dir_y):
    h, w = img.shape
    sum = 0
    for x in range(w-2):
        for y in range(h-2):
            sum = sum + np.square(img[y,x] - img[y-dir_y, x-dir_x] - img[y+dir_y, x+dir_x])
    return sum/(h*w)

# sobel 算子以去除横竖栅栏
def sobel_fun(img_gauss):
    imgs = []
    for i in range(len(img_gauss)):

        sobel_x = cv2.Sobel(img_gauss[i],cv2.CV_16S,1,0)
        sobel_y = cv2.Sobel(img_gauss[i],cv2.CV_16S,0,1)

        # sobel_x = cv2.Sobel(img,cv2.CV_16S,1,0)
        # sobel_y = cv2.Sobel(sobel_x,cv2.CV_16S,0,1)

        # return cv2.convertScaleAbs(sobel_y)

        # print(td(sobel_y, 0, 1)-td(sobel_x, 1, 0))
        
        if td(sobel_y, 0, 1)-td(sobel_x, 1, 0) > 40:
            # print('sobel_x')
            # return cv2.convertScaleAbs(sobel_y)
            # io.imshow(cv2.convertScaleAbs(sobel_x))
            # io.show()
            # io.imshow(cv2.convertScaleAbs(sobel_y))
            # io.show()
            # io.imshow(cv2.convertScaleAbs(sobel_y))
            # io.show()
            img = cv2.convertScaleAbs(sobel_x)
             
        elif td(sobel_x, 1, 0)-td(sobel_y, 0, 1) > 40:
            # print('sobel_y')
            # return cv2.convertScaleAbs(sobel_x)   # 转回uint8
            # io.imshow(cv2.convertScaleAbs(sobel_y))
            # io.show() 
            # sobel_xx = cv2.Sobel(sobel_y,cv2.CV_16S,1,0)
            img = cv2.convertScaleAbs(sobel_y)
            # return img
            
        else :
            # print('else')
            # return cv2.convertScaleAbs(sobel_x)
            img = cv2.convertScaleAbs(sobel_x)
        imgs.append(img)
    return imgs


def test():
    emm = cv2.imread('/Users/ryshen/Desktop/test.png')
    img = cv2.imread('/Users/ryshen/Desktop/test.png', 0)
    img1 = img.astype('float')
    # dst = color.rgb2gary(img)
    # img1 = io.imread(image_list[2], 1)
    # print(img1.dtype)
    # io.imshow(dst)
    # io.show()  

    img_h, img_w = img1.shape 
    # 为对中心进行汉宁窗需要两个hann数组
    hann =  signal.hann(img_w)
    hann2 = signal.hann(img_h)
    img_hann = (img1.T * hann2).T * hann

    square = np.square(img_hann) 
    gauss = filters.gaussian_filter(square,0.005)
    normalizedImg = np.zeros((720, 1280))
    img_gauss = cv2.normalize(gauss,  normalizedImg, 0, 255, cv2.NORM_MINMAX)


    sobel_x = cv2.Sobel(img_gauss,cv2.CV_16S,1,0)
    sobel_y = cv2.Sobel(img_gauss,cv2.CV_16S,0,1)

    
    if td(sobel_y, 0, 1)-td(sobel_x, 1, 0) > 40:

        img_sobel = cv2.convertScaleAbs(sobel_x)
         
    elif td(sobel_x, 1, 0)-td(sobel_y, 0, 1) > 40:

        img_sobel = cv2.convertScaleAbs(sobel_y)
       
    else :
        img_sobel = cv2.convertScaleAbs(sobel_x)


    kernel_open = np.ones((5, 5), np.uint8)
    kernel_dilation = np.ones((15, 15), np.uint8)

    ret,thresh=cv2.threshold(img_sobel,50,255,cv2.THRESH_BINARY) 

    img_open = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel_open)
    img_dilation = cv2.dilate(img_open,kernel_dilation,iterations = 1)

    bin8bit = img_dilation.astype(np.uint8)
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

            # plate_contour_minRectangle = cv2.minAreaRect(contours[j])
            # plate_points = cv2.boxPoints(plate_contour_minRectangle).astype(int)
            # print(plate_points)   
    # 用画矩形方法绘制正外接矩形
    # print(max_rect)
    # print(cv2imgs_origin[i])
    # green = cv2.rectangle(emm, (int(max_rect[0]*0.9), int(max_rect[1]*0.9)), (max_rect[0]+int(max_rect[2]*1.2), max_rect[1]+int(max_rect[3]*1.2)), (0, 255, 0), 3);
    # io.imshow(img_dilation)
    # # # io.imshow(green)
    # io.show() 
    # green = cv2.rectangle(cv2imgs_origin[i], (int(max_rect[0]*0.9), int(max_rect[1]*0.9)), (max_rect[0]+int(max_rect[2]*1.2), max_rect[1]+int(max_rect[3]*1.2)), (0, 255, 0), 3);
    green = cv2.rectangle(emm, (int(max_rect[0]), int(max_rect[1])), (max_rect[0]+int(max_rect[2]), max_rect[1]+int(max_rect[3])), (0, 255, 0), 3);





    plt.subplot(231)
    plt.imshow(img,'gray')
    plt.title('gray')  
    plt.subplot(232)
    plt.imshow(img_hann,'gray')
    plt.title('hanning')
    plt.subplot(234)
    plt.imshow(img_gauss,'gray')
    plt.title('gauss')
    plt.subplot(235)
    plt.imshow(img_sobel,'gray')
    plt.title('sobel')
    # plt.subplot(235)
    # plt.imshow(img_dilation,'gray')
    # plt.title('dilation')
    # plt.show()
    plt.subplot(236)
    plt.imshow(green,'gray')
    plt.title('final')
    plt.show()

    # cv2.imshow('image3',img)
    # cv2.waitKey(0)
    # cv2.imshow('image2',img1)
    # cv2.waitKey(0)
    # cv2.imshow('image',img_recor)
    # cv2.waitKey(0)
    # cv2.imshow('image1',img_dct)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




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

# test()


