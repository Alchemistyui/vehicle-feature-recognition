import numpy as np
import cv2
import os

LOWER_BLUE = np.array([80, 70, 60])
UPPER_BLUE = np.array([120, 255, 255])
CLOSE_KERNEL = np.ones((4, 18), np.uint8)
CLOSE_KERNEL2 = np.ones((5, 20), np.uint8)
OPEN_KERNEL = np.ones((5, 5), np.uint8)
OPEN_KERNEL2 = np.ones((5, 16), np.uint8)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
MIN_SQUARE = 2200
MAX_SQUARE = 25000
MAX_ANGLE = 30
MIN_PROPORTION = 2
MAX_PROPORTION = 6
MIN_BLUE_PROPORTION = 0.4

path = r'/Users/ryshen/Desktop/车辆' #文件夹目录
_dir = 0
_file = 0
_none = 0
_site = 0
name = 0
def Load_Img(path):
    global _dir,_file,_none,_site, name
    
    files = os.listdir(path)

    for file in files: #遍历文件夹
        if os.path.isdir(path+'/'+file): #判断是否是文件夹，不是文件夹才打开
            _dir += 1
            Load_Img(path+'/'+file)
            
        else :
            img = cv2.imdecode(np.fromfile(path+'/'+file, dtype=np.uint8), -1)
            if img is None:
                _none += 1
            else:
                img_Gaussianblur = cv2.GaussianBlur(img,(3, 3),0)
                #cv2.imshow("Gaussian blur",img_Gaussianblur)
                #cv2.waitKey(0)
                
                img_HSV = cv2.cvtColor(img_Gaussianblur, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(img_HSV, LOWER_BLUE, UPPER_BLUE)
                #cv2.imshow("HSV scale",mask)
                #cv2.waitKey(0)
                
                #img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
                img_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)
                img_open1 = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, OPEN_KERNEL)
                img_close1 = cv2.morphologyEx(img_open1, cv2.MORPH_CLOSE, CLOSE_KERNEL2)
                img_open2 = cv2.morphologyEx(img_close1, cv2.MORPH_OPEN, OPEN_KERNEL2)
                img_close2 = cv2.morphologyEx(img_open2, cv2.MORPH_CLOSE, CLOSE_KERNEL2)

                #cv2.imshow("morphological open1 operate", img_open2)
                #cv2.waitKey(0)
                
                #cv2.imshow("morphological open2 operate", img_close2)
                #cv2.waitKey(0)
                

                img_ret, contours, hierarchy = cv2.findContours(img_close2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_contours = cv2.drawContours(img.copy(), contours, -1, RED, 2)
                #cv2.imshow("Contours", img_contours)
                #cv2.waitKey(0)

                img_copy = img.copy()
                for contour in contours:
                    if (Judge_Contour_Size(contour,img_copy) is not False):
                        img_contour_plate = cv2.drawContours(img.copy(), contour, -1, RED, 2)
                        #cv2.imshow("Contour_plate", img_contour_plate)
                        #cv2.waitKey(0)
                        if (Judge_Contour_Color(contour,img_copy) is not False):
                            img_plate_contours = cv2.drawContours(img.copy(), contour, -1, GREEN, 2)
                            #cv2.imshow("Plate_Contours", img_plate_contours)
                            #cv2.waitKey(0)

                             # 彪哥加的代码
                            max_rect = cv2.boundingRect(contour)
                            start = 0
                            if max_rect[1]-2*max_rect[3] > 0:
                                start = int(max_rect[1]-2*max_rect[3])
                            # cutImg = img[max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]
                            cutImg = img[start:int(start+max_rect[3]*2 ), max_rect[0]:int(max_rect[0]+max_rect[2])]
                            name = name + 1
                            cv2.imwrite('/Users/ryshen/Desktop/粗定位/'+str(name)+'.png', cutImg)
                            # print(path+'/'+file)

                            # plate_contour_minRectangle = cv2.minAreaRect(contour)
                            # plate_points = cv2.boxPoints(plate_contour_minRectangle).astype(int)
                            # print(plate_points)                           #---------------------------------四元组返回入口

                            _site += 1
                            
                            #cv2.destroyAllWindows()
                            break
            
                cv2.destroyAllWindows()
            
            _file += 1

def Judge_Contour_Size(contour,img):
    if contour is None:
        return False
    
    #提取最小矩形和矩形四元组等属性
    contour_minRectangle = cv2.minAreaRect(contour)
    points = cv2.boxPoints(contour_minRectangle).astype(int)
    
    width, height, angle = contour_minRectangle[1][0], contour_minRectangle[1][1], contour_minRectangle[2]
    img_width, img_height = img.shape[:2]
    
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
        return False

    #print("square: %.5f  prop: %.5f  angle: %d" % (square, proportion, angle))

    if MIN_SQUARE < square < MAX_SQUARE and MIN_PROPORTION < proportion < MAX_PROPORTION and abs(angle) < MAX_ANGLE:
        return True
    else:
        return False

def Judge_Contour_Color(contour, img_copy):
    if contour is None:
        return False

    contour_minRectangle = cv2.minAreaRect(contour)
    points = cv2.boxPoints(contour_minRectangle).astype(int)

    img_width, img_height = img_copy.shape[:2]
    
    for point in points:
        if point[0] > img_width or point[0] < 0:
            if point[1] > img_height or point[1] < 0:
                return False
            
    center_x, center_y = contour_minRectangle[0][0], contour_minRectangle[0][1]
    width, height, angle = contour_minRectangle[1][0], contour_minRectangle[1][1], contour_minRectangle[2]

    width, height = int(width),int(height)

    contour_x, contour_y = int(center_x), int(center_y)
    #print("center:(", contour_x, ",", contour_y, ")")

    deal = cv2.getRotationMatrix2D((contour_x, contour_y), angle, 1)
    img_rotated = cv2.warpAffine(img_copy, deal, (img_height, img_width))
    contour_rotated = img_rotated[int(contour_y - height/2):int(contour_y + height/2), int(contour_x - width/2):int(contour_x + width/2)]
    
    if height < 0 or width < 0:
        return False
    
    #cv2.imshow("rotated",contour_rotated)
    #cv2.waitKey(0)
    
    #如果矩形有端点超出图像，则该判定该矩形不是需要的车牌
    if contour_x + width/2 > img_width or contour_x - width/2 < 0:
        if contour_y + height/2 > img_height or contour_y - height/2 < 0:
            return False
    
    rotated_HSV = cv2.cvtColor(contour_rotated, cv2.COLOR_BGR2HSV)

    if rotated_HSV is None:
        return False
    
    mask = cv2.inRange(rotated_HSV, LOWER_BLUE, UPPER_BLUE)

    total = mask.size
    blue = mask[mask == 255].size

    if blue / total < MIN_BLUE_PROPORTION:
        return False
    else:
        print(_file ,"-------blue / total = ",blue / total)
        return True
    
if __name__ == "__main__":
    Load_Img(path)
    print(_dir,_file,_none,_site)
