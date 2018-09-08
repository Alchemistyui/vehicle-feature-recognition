# mnist数据源
# Sequential类，可以封装各种神经网络层，包括Dense全连接层，Dropout层，Cov2D 卷积层等
# keras后端TensorFlow

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# import numpy as np
# import matplotlib.pyplot as plt

# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Dropout
# from keras.utils import np_utils


# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。
# batch_size = 128
# 0-9手写数字一个有10个类别
num_classes = 2
# epochs,12次完整迭代
# epochs = 12
# 输入的图片是28*28像素的灰度图
img_rows, img_cols = 250, 250
# 训练集，测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# # keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，
if K.image_data_format() == 'channels_first':
 x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
 x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
 input_shape = (3, img_rows, img_cols)
else:
    # x_train(所有图像，1灰度通道，行，列)
 x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
 x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
 input_shape = (img_rows, img_cols, 3)

# # 二维数据变成一维数据
# x_train = x_train.reshape(len(x_train), -1)
# x_test = x_test.reshape(len(x_test), -1)


# uint不能有负数，先转为float类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 数据归一化,减去均值除以范围,最终是0-1的范围,
# 所以最后的激活函数应该是sigmoid,如果是-1~1,那么激活函数应该是tanh
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 把类别0-9变成2进制，方便训练
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)



for i in range(1,9):
    # Sequential类
    model = Sequential()

    # 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
    # 卷积核的窗口选用3*3像素窗口
    # 二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，
    # 应提供input_shape参数。例如input_shape = (128,128,3)代表128*128的彩色RGB图像
    # filters即输出的维度，不用考虑输入维度，也就是卷积核的数目，也就是将平面的图像，拉伸成filters维的空间矩阵
    # nb = 30+20*i
    core = 30+2*i
    step = 2*i
    model.add(Conv2D(16,
     kernel_size= core,
     strides=step,
     activation='relu',
     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 64个通道的卷积层
    # 替代全连接层
    model.add(Conv2D(400,
     kernel_size= 5,
     activation='relu'))
    model.add(Conv2D(400,
     kernel_size= 1,
     activation='relu'))
    model.add(Conv2D(1,
     kernel_size= 1,
     activation='relu'))
    # 对输入采用0.5概率的Dropout
    model.add(Dropout(0.5))
    # 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
    model.add(Dense(num_classes, activation='softmax'))




    # 配置模型使用交叉熵损失函数，最优化方法选用Adadelta
    # compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
    # optimizer：字符串（预定义优化器名）或优化器对象，参考优化器
    # loss：字符串（预定义损失函数名）或目标函数，参考损失函数
    # metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练过程
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))

    # x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
    # y：标签，numpy array
    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    # epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。
    # validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。

    model.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model'+str(i)+'.h5')
    del model



