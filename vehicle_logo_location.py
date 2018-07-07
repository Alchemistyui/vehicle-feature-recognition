import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2

# 二、导入 hdf5 数据集
#我的图片大小为（64*64*3）
train_dir = '/Users/ryshen/Desktop'
train_dataset = h5py.File(train_dir + 'data.h5', 'r')
X_train = np.array(train_dataset['X_train'][:]) # your train set features
X_test = np.array(train_dataset['X_test'][:]) # your train set features
train_dataset.close()


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("X_test shape: " + str(X_test.shape))

# plt.imshow(X_train[2])
# plt.show()

img = cv2.imread(X_train[2],cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



