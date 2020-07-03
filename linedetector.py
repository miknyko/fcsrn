import cv2
import numpy as np
import math

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

class LineDetector():
    def __init__(self,img_path):
        self.image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        self.thresh_holds = [100,70,50,40]

    def preprocess(self):
        """preprocess image ----gray,blur,and find edges"""
        self.image_copy = self.image.copy()
        self.gray = cv2.cvtColor(self.image_copy,cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray,(3,3),0)
        self.edges = cv2.Canny(self.gray,50,150,apertureSize=3)

    def houghlines(self):
        """find lines using HoughLines """
        self.preprocess()
        for thresh in self.thresh_holds:
            self.lines = cv2.HoughLines(self.edges,1,np.pi/180,thresh)
            if self.lines is not None:
                self.lines_quantity = len(self.lines)
                print(f'[INFO] thresh hold is {thresh},find {self.lines_quantity} lines')
                break
            

    def draw_lines(self):
        """draw all the lines detected"""
        self.angle_sum = 0
        for line in self.lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            # change to x-y coord system
            x0 = a * rho
            y0 = b * rho
            # get coords of 2 points on the same line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(self.image_copy,(x1,y1),(x2,y2),(0,0,255),1)
            if x1 == x2 or y1 == y2:
                continue
            self.angle_sum += math.degrees(math.atan(float((y2 - y1) / (x2 - x1))))
            
        self.angle_average = self.angle_sum / len(self.lines)
        # cv2.imshow('alllines',self.image_copy)
        # cv2.waitKey()
        # cv2.destoryAllwindows()


    def rotate(self):
        if self.angle_average > 45:
            self.angle_average = -90 + self.angle_average
        elif self.angle_average < -45:
            self.angle_average = 90 + self.angle_average
        self.rotate_image = rotate_bound(self.image_copy,-self.angle_average)
        cv2.imshow('rotate_test',self.rotate_image)
        cv2.waitKey()
        cv2.destoryAllwindows()
        

    
if __name__ == "__main__":
    test_image = r'fcsrn_train_images/croped_微信图片_2019120910243222.jpg'
    detecs = LineDetector(test_image)
    detecs.houghlines()
    detecs.draw_lines()
    detecs.rotate()
    

        
        


        