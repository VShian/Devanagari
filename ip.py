import cv2
import numpy as np
# from matplotlib import pyplot as plt

#ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#hist = cv2.calcHist([thresh],[0],None,[2],[0,2])
# height,width = img.shape
# print thresh.shape

def get_chars(start,end, img):
	height, width = img.shape

	for i in range(start, end-1):
		if sum(img[i]) <= 0.3*width and sum(img[i+1]) >= 0.5*width:
			start=i+1
			break

	req_height = end-start
	
	m_t = np.matrix(img).T

	flag = 0
	chars = []
	i=0
	for i in range(width-1):
		if np.sum(m_t[i,start:end]) >= req_height and np.sum(m_t[i+1,start:end]) < req_height and flag == 0:
			chars+=[[i+1,-1]]
			flag = 1
		elif np.sum(m_t[i,start:end]) < req_height and np.sum(m_t[i+1,start:end]) >= req_height and flag == 1:
			chars[-1][1] = i+1
			flag = 0
	if len(chars)!=0 and chars[-1][1]==-1:
		chars[-1][1]=i
	return chars


def get_lines(img):
	height,width = img.shape

	lines = []
	flag,i = 0,0
	for i in range(height-1):
		if sum(img[i]) >= 0.95*width and sum(img[i+1]) < 0.95*width and flag == 0:
			lines+=[[i+1,-1]]
			flag = 1
		elif sum(img[i]) < 0.95*width and sum(img[i+1]) >= 0.95*width and flag == 1:
			lines[-1][1]=i+1
			flag = 0
	if len(lines)!=0 and lines[-1][1]==-1:
		lines[-1][1]=i
	return lines

# cv2.imshow("cropped", resized_img)
# cv2.waitKey(0)