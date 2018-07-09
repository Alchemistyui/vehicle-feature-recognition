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
# import pandas as pd
# from sklearn import preprocessing

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
    # print(train_set_x_orig.shape)
    # print(test_set_x_orig.shape)
    #测试
    # plt.imshow(train_set_x_orig[2])
    # pylab.show()

def load_picture(image_list):
    imgs = [];
    for i in range(len(image_list)):
        img = cv2.imread(image_list[i], 0)
        # dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
    # 测试
    # io.imshow(imgs[2])
    # io.show()
    # for i in range(len(imgs)):
    #     cv2.imshow('image',imgs[i])
    #     cv2.waitKey(5)
    #     cv2.destroyAllWindows()
    return imgs


def dct(cv2imgs):
    imgs = []
    # imgs_log = []
    for i in range(len(cv2imgs)):
        img = cv2imgs[i].astype('float')
        img_dct = cv2.dct(img)    
        # img_dct_log = np.log(abs(img_dct))  #进行log处理
        imgs.append(img_dct)
        # imgs_log.append(img_dct_log)
    # 测试
    # plt.imshow(imgs_log[2], 'gray')
    # plt.title('DCT')
    # plt.show()
    # io.imshow(cv2imgs[2].astype('float'))
    # io.show()
    # io.imshow(imgs[2])
    # io.show()
    # cv2.imshow('image2',cv2imgs[2].astype('float'))
    # cv2.waitKey(0)
    # for i in range(len(cv2imgs)):
    #     cv2.imshow('image',imgs[i])
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows()
    return imgs

def idct(img_dct):
    imgs = [];
    for i in range(len(img_dct)):
        img_recor = cv2.idct(img_dct[i])
        imgs.append(img_recor) 
    # 测试
    # test = imgs[2].astype()
    # io.imshow(imgs[2])
    # io.show()

    # plt.imshow(imgs[2],'gray')
    # plt.title('IDCT2(cv2_idct)')
    # plt.show()
    # cv2.imshow('image2',imgs[2])
    # cv2.waitKey(0)
    # for i in range(len(img_dct)):
    #     cv2.imshow('image',imgs[i])
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows()
    return imgs

def sign(img_dct_log):
    imgs = []
    for i in range(len(img_dct_log)):
        sign = np.sign(img_dct_log[i])
        imgs.append(sign)
    # io.imshow(imgs[2])
    # io.show()
    # plt.imshow(imgs[2],'gray')
    # plt.title('sign(cv2_idct)')
    # plt.show()
    return imgs



def test(cv2imgs):
    img = cv2.imread('/Users/ryshen/Desktop/test4.png', 0)
    img1 = img.astype('float')
    # dst = color.rgb2gary(img)
    # img1 = io.imread(image_list[2], 1)
    # print(img1.dtype)
    # io.imshow(dst)
    # io.show()  
    hann = signal.hann(420)
    img2 = img1 ** hann
    # pl.plot(hann)
    # pl.show()
    # scipy.signal.hamming()
    # img2 = img1.rolling(window=5, win_type='hamming')

    img_dct = cv2.dct(img2)         #进行离散余弦变换
     
    # img_dct_log = np.log(abs(img_dct))  #进行log处理
    # sign = np.sign(img_dct) 
    # print(img_dct)
    # print(sign)
    # sign = np.where(np.absolute(img_dct)<90, 0, img_dct)
    sign = np.where(img_dct<20, -50, img_dct)

    # square = np.square(sign)
    # gauss = filters.gaussian_filter(square,0.01)

    
    img_recor = cv2.idct(sign)    #进行离散余弦反变换

    square = np.square(img_recor) 
    # power = np.power(img_recor, 5)
    # power = img_recor * 10
    gauss = filters.gaussian_filter(square,0.005)

    # normal = Normalize(gauss)
    # normal = preprocessing.MinMaxScaler()

    normalizedImg = np.zeros((720, 1280))
    normalizedImg = cv2.normalize(gauss,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

    # normal = normalizedImg * 255
    # print(sign)

    # print(img_recor)


    # normal=exposure.rescale_intensity(img_recor,'img_recor', intensity_range(0, 255))

    # print(gauss)

    # print(normalizedImg)

    # enh_con = ImageEnhance.Contrast(img_recor)  
    # contrast = 1.5  
    # img_contrasted = enh_con.enhance(contrast) 

    # io.imshow(img_dct)
    # io.show()
    # io.imshow(img_dct_log)
    # io.show()
    # io.imshow(img_recor)
    # io.show()

    # result = cv2.medianBlur(img_recor,5)



    ret,thresh=cv2.threshold(gauss,127,255,cv2.THRESH_BINARY) 

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_erosion = np.ones((9, 9), np.uint8)
    kernel_dilation = np.ones((13, 13), np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    img_open = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    # img_open = cv2.morphologyEx(gauss,cv2.MORPH_OPEN,kernel_open)
    img_erosion = cv2.erode(img_open,kernel,iterations = 1)
    img_dilation = cv2.dilate(img_erosion,kernel,iterations = 1)

    # img_close = cv2.morphologyEx(gauss,cv2.MORPH_CLOSE,kernel_open)


    ret, binary = cv2.threshold(img_dilation,127,255,cv2.THRESH_BINARY)
    # gray = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY)
    bin8bit = binary.astype(np.uint8)
    # bin8bit = thresh.astype(np.uint8)


    ret, contours, hierarchy = cv2.findContours(bin8bit,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    test = cv2.imread('/Users/ryshen/Desktop/test4.png')
    # img_contours = cv2.drawContours(test,contours,-1,(255,0,0),3)

    max_size = 0
    max_rect = [0, 0, 0, 0]

    for i in range(len(contours)):
        # //2.4 由轮廓（点集）确定出正外接矩形并绘制
        # boundRect[i] = boundingRect(Mat(contours[i]));
        # # //2.4.1获得正外接矩形的左上角坐标及宽高  
        # int width = boundRect[i].width;
        # int height = boundRect[i].height;
        # int x = boundRect[i].x;
        # int y = boundRect[i].y;
        # //2.4.2用画矩形方法绘制正外接矩形
        x, y, w, h = cv2.boundingRect(contours[i])
        if w*h > max_size:
            max_rect[0] = x 
            max_rect[1] = y
            max_rect[2] = w
            max_rect[3] = h
    # print(max_rect)
    # green = cv2.rectangle(test, (max_rect[0], max_rect[1]), (max_rect[0]+max_rect[2], max_rect[1]+max_rect[3]), (0, 255, 0), 5);
    # green = cv2.drawContours(test,max_rect,-1,(0, 255,0),3)
    cutImg = test[max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]

    # plt.subplot(236)
    # plt.imshow(cutImg)
    # plt.title('cutImg')
    # plt.show()



    plt.subplot(331)
    plt.imshow(img,'gray')
    plt.title('original')  
      
    plt.subplot(332)
    plt.imshow(img2,'gray')
    plt.title('hamming')

    
    plt.subplot(333)
    plt.imshow(green,'gray')
    plt.title('green')
    plt.subplot(334)
    plt.imshow(sign,'gray')
    plt.title('sign')
    plt.subplot(335)
    plt.imshow(img_recor,'gray')
    plt.title('idct')
    # plt.show()
    plt.subplot(336)   
    plt.imshow(gauss,'gray')
    plt.title('gauss')
    plt.subplot(337)   
    plt.imshow(thresh,'gray')
    plt.title('thresh')
    plt.subplot(338)   
    plt.imshow(img_open,'gray')
    plt.title('img_open')
    plt.subplot(339)   
    plt.imshow(img_erosion,'gray')
    plt.title('img_erosion')
    # plt.subplot(337)  
    # plt.imshow(img_dilation,'gray')
    # plt.title('img_dilation')

    # plt.imshow(img_dilation,'gray')
    # plt.title('img_dilation')
    
    
    # plt.subplot(336)
    # plt.imshow(img_contours,'gray')
    # plt.title('img_contours')
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


def Gaussian(sign_img):
    gauss = []
    for i in range(len(sign_img)):
        square = np.square(sign_img[i])
        img = filters.gaussian_filter(square,0.05)
        gauss.append(img)
    return gauss

def Normalize(data):
    # imgs = []
    # for i in range(len(data)):
    m = np.mean(data)
    mx = data.max()
    mn = data.min()
    return ((float(j) - m) / (mx - mn) * 255 for j in data)
        # imgs.append((float(j) - m) / (mx - mn) * 255 for j in data)


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
# img_dct = dct(cv2imgs)
# sign_img = sign(img_dct)
# # gauss_img = Gaussian(sign_img)
# # normal = Normalize(gauss_img)
# idct_img = idct(sign_img)
test(cv2imgs)
# print(len(label_list))





