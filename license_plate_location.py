import numpy as np
import cv2
import os


SOBEL = 3
OPEN_KERNEL = np.ones((3, 15), np.uint8)
CLOSE_KERNEL = np.ones((4, 18), np.uint8)
THICKNESS = 2
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
MIN_SQUARE = 3000
MAX_SQUARE = 22000
MAX_ANGLE = 30
MIN_PROPORTION = 2
MAX_PROPORTION = 7
LOWER_BLUE = np.array([100, 43, 46])
UPPER_BLUE = np.array([120, 255, 255])
MIN_BLUE_PROPORTION = 0.2
imgs = []
success = 0
path = "/Users/ryshen/Desktop/车辆" #文件夹目录
name = 0
cutImgs = []

def Load_Img(path):
    global imgs
    
    files= os.listdir(path) #得到文件夹下的所有文件名称
    
    # print(files)
    for file in files: #遍历文件夹
        # print(path+file)
        # print(os.path.isdir(path+file))
        if os.path.isdir(path+'/'+file): #判断是否是文件夹，不是文件夹才打开
            Load_Img(path+'/'+file)

        else :
            # print(os.path.isdir(file))
            # img = cv2.imread(path+'/'+file, 0)
            img_original = cv2.imdecode(np.fromfile(path+'/'+file, dtype=np.uint8), -1)
            # print("append")
            imgs.append(img_original) 
    # (img[0]) #打印结果
    
def Batch_Location():
    for i in range(len(imgs)):
        Location(imgs[i])



def Location(img_original):
    global success, cutImgs, name
    # img_original = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)





    #高斯模糊图片
    # print("--高斯模糊中--")
    img_Gaussianblur = cv2.GaussianBlur(img_original,(5, 5),0)
    #cv2.imshow("Gaussian blur",img_Gaussianblur)
    #cv2.waitKey(0)

    #灰度化图片
    # print("--灰度化中--")
    img_gray = cv2.cvtColor(img_Gaussianblur,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray scale",img_gray)
    # cv2.waitKey(0)







    #提取垂直方向边缘
    # print("--边缘提取中--")
    # img_edge = cv2.Sobel(img_Gaussianblur, cv2.CV_64F, 1, 0, SOBEL)
    img_edge = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, SOBEL)

    img_edge_abs = cv2.convertScaleAbs(img_edge)
    # cv2.imshow("edge", img_edge_abs)
    # cv2.waitKey(0)

    #二值化图片
    # print("--二值化中--")
    ret, img_thresh = cv2.threshold(img_edge_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("binary image", img_thresh)
    # cv2.waitKey(0)

    #开闭操作
    # print("--开闭操作中--")
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, OPEN_KERNEL)
    
    # cv2.imshow("morphological close operate", img_open)
    # cv2.waitKey(0)

    #轮廓检测并画出来
    print("--轮廓检测中--")
    img_ret, contours, hierarchy = cv2.findContours(img_open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = cv2.drawContours(img_original.copy(), contours, -1, BLUE, THICKNESS)
    # cv2.imshow("Contours", img_contours)
    # cv2.waitKey(0)

    img_copy = img_original.copy()
    
    #对每一个轮廓取最小矩形，并根据矩形形状和颜色来判断是否为车牌
    for contour in contours:
        if Judge_Contour_Size(contour,img_copy) and Judge_Contour_Color(contour, img_copy):
            
            #如果矩形的形状和颜色都判断成功，则画出对应的轮廓并显示
            max_rect = cv2.boundingRect(contour)
            print(max_rect)
            my_contour = cv2.rectangle(img_original.copy(), (int(max_rect[0]), int(max_rect[1])), (max_rect[0]+int(max_rect[2]), max_rect[1]+int(max_rect[3])), (0, 255, 0), 3);
            #  (x, y, w, h) 
            # cutImg = img_original[max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]
            # (x, y- 3* height, 3* height, width)
            cutImg = img_original[max_rect[1]-4*max_rect[3]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]
            cutImgs.append(cutImg)
            cv2.imwrite('/Users/ryshen/Desktop/粗定位/'+str(name)+'.png', cutImg)
            name = name + 1
            # print(name)
            # cv2.imshow("Contours", cutImg)
            # cv2.waitKey(0)
            
            #取出该轮廓的最小矩形并返回它的四元组
            contour_minRectangle = cv2.minAreaRect(contour)
            points = cv2.boxPoints(contour_minRectangle).astype(int)
            
            success = success + 1
            print(points)
            return points
    # cv2.imshow("Contours", cutImgs[0])
    # cv2.waitKey(0)



def Judge_Contour_Size(contour,img_copy):
    
    #提取最小矩形和矩形四元组等属性
    contour_minRectangle = cv2.minAreaRect(contour)
    points = cv2.boxPoints(contour_minRectangle).astype(int).reshape(-1, 2)
    
    width, height, angle = contour_minRectangle[1][0], contour_minRectangle[1][1], contour_minRectangle[2]
    img_width, img_height = img_copy.shape[:2]
    
    #如果矩形有端点超出图像，则该判定该矩形不是需要的车牌
    for point in points:
        if point[0] > img_width or point[0] < 0:
            if point[1] > img_height or point[1] < 0:
                return False
        
    #定向的让width大于height，方便后面长宽比的计算
    if height > width:
        width, height = height, width
        angle += 90

    #整型化长宽的值
    width = int(width)
    height = int(height)

    #计算矩形的面积和长宽比，若面积在最大和最小面积之间，并且长宽比也在所需的范围内，则形状判定成功，否则判定失败
    square = width * height

    if height > 0:
        proportion = width / height
    else:
        proportion = float("Inf")

    if MIN_SQUARE < square < MAX_SQUARE and MIN_PROPORTION < proportion < MAX_PROPORTION and abs(angle) < MAX_ANGLE:
        return True
    else:
        return False

def Judge_Contour_Color(contour, img_copy):

    global contour_x, contour_y

    #取最小矩形和矩形四元组等属性
    contour_minRectangle = cv2.minAreaRect(contour)
    points = cv2.boxPoints(contour_minRectangle).astype(int).reshape(-1, 2)

    center_x, center_y, width, height, angle = contour_minRectangle[0][0], contour_minRectangle[0][1], contour_minRectangle[1][0], contour_minRectangle[1][1], contour_minRectangle[2]

    width, height = int(width),int(height)

    #取出顶点中最接近原点的一个，以此点作为后面进行旋转扭正矩形的旋转中心（可以不唯一）
    for point in points:
        if point[0] < center_x and point[1] < center_y:
            contour_x, contour_y = int(point[0]), int(point[1])
            break

    img_width, img_height = img_copy.shape[:2]

    #把有倾斜角的矩形旋转扭正为与坐标轴平行的矩形，并把矩形所在图像的部分取出来，方便计算矩形的像素点
    deal = cv2.getRotationMatrix2D((contour_x, contour_y), angle, 1)
    img_rotated = cv2.warpAffine(img_copy, deal, (img_height, img_width))
    contour_rotated = img_rotated[contour_y:contour_y + height, contour_x:contour_x + width]

    #转化为HSV图像，并加上掩膜，只允许在要求范围内的颜色保存下来
    contour_HSV = cv2.cvtColor(contour_rotated, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(contour_HSV, LOWER_BLUE, UPPER_BLUE)
    

    #计算矩形中蓝色的像素占总的矩形像素块的比例，根据比例是否在阈值内来判定该矩形是否是车牌
    total_area = width * height
    blue_area = mask[mask == 255].size

    blue_porprotion = blue_area / total_area

    if blue_porprotion > MIN_BLUE_PROPORTION:
        return True
    else:
        return False




Load_Img(path)
print(len(imgs))
Batch_Location()

print('success : ')
print(success)
print('accuracy rate : ')
print(success/len(imgs))

# print(imgs)
# cv2.imshow("Gaussian blur",imgs[0])
# cv2.waitKey(10)

# if __name__ =="__main__":
img = r"/Users/ryshen/Desktop/test.jpg"
    # img = r"C:\Users\Zelinger\Desktop\1970_01_01_08_02_35_262861_川AFQ892_蓝牌img0.jpg")
    #img = r"C:\Users\Zelinger\Desktop\1970_01_25_21_33_21_113934_川A0B002_蓝牌img0.jpg"
    #img = r"C:\Users\Zelinger\Desktop\1970_01_14_01_48_22_703382_川A9PE08_蓝牌img0.jpg"
    #img = r"C:\Users\Zelinger\Desktop\1970_01_31_03_25_10_237612_川A1N1T8_蓝牌img0.jpg"
    
# Location(img)
