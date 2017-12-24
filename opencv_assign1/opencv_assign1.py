import numpy as np
import cv2

#Color Correction
#1
img1 = cv2.imread('input/GUC.png',0)
out1 = np.where(img1 < 50,0,img1-50)
#cv2.imshow('image1',out1)
cv2.imwrite('output/GUC_out.png',out1)
#2
img2 = cv2.imread('input/calculator.png',0)
out2 = np.where(img2 > 185 ,255,img2+70)
out2=out2-70
#cv2.imshow('image2',out2)
cv2.imwrite('output/calculator_out.png',out2)
#3
img3 = cv2.imread('input/cameraman.png',0)
out3 = np.where(img3 < 30 ,img3*3,img3)
#cv2.imshow('image3',out3)
cv2.imwrite('output/cameraman_out.png',out3)
#Basic Segmentation
img4 = cv2.imread('input/lake.png',0)
img4_temp=cv2.Canny(img4,100,200)
img4_temp = cv2.adaptiveThreshold(img4_temp,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
img4_temp=255-img4_temp
out4 = cv2.bitwise_and(img4,img4_temp)
#cv2.imshow('image4',out4)
cv2.imwrite('output/lake_out.png',out4)
#Images Combination
#1
img5 = cv2.imread('input/james.png',0)
img5_temp = np.where(img5==0,img5+1,img5)
rows,cols = img5_temp.shape
M = np.float32([[1,0,-200],[0,1,0]])
img5_translated = cv2.warpAffine(img5_temp,M,(cols,rows))
img5_translated = np.where(img5_translated==0,255,img5_translated)
img5_translated = np.where(img5_translated>240,255,img5_translated)
img6 = cv2.imread('input/london1.png',0)
out5= np.where(img5_translated==255,img6,img5_translated)
#cv2.imshow('image5',out5)
cv2.imwrite('output/london1_out.png',out5)
#2
img5_flipped = cv2.flip(img5,1)
img5_flipped = np.where(img5_flipped>240,255,img5_flipped)
img7 = cv2.imread('input/london2.png',0)
out6= np.where(img5_flipped==255,img7,img5_flipped)
#cv2.imshow('image6',out6)
cv2.imwrite('output/london2_out.png',out6)

#cv2.waitKey(0)
#cv2.destroyAllWindows()