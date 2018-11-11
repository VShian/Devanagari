import cv2
import numpy as np
# from matplotlib import pyplot as plt
import ip
'''
import nn

(train_images, train_labels), (test_images,test_labels) = load_data()
model = load_model(len(train_images[1]))
history = model.fit(train_images, train_labels, epochs=5, batch_size=256, verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print test_loss, test_acc
def predict(img):
	print model.predict(img.flatten())
'''

img = cv2.imread('a.jpg',cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img,(5,5),0)
ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
height,width = img.shape

bin_img = np.zeros((img.shape),dtype='float32')

'''
for i in range(height):
	print sum(thresh[i])
'''


for i in range(height):
	for j in range(width):
		if img[i][j] >=127:
			bin_img[i][j]=1
		else:
			bin_img[i][j]=0

lines = ip.get_lines(bin_img)

for start,end in lines:
	chars = ip.get_chars(start,end,bin_img)
	for char in chars:
		character = np.array(thresh[start:end, char[0]:char[1]])
		resized_img = cv2.resize(character,(32,32))
		cv2.imshow('gray',resized_img)
		cv2.waitKey(0)
		break