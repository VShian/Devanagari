import cv2
import numpy as np
# from matplotlib import pyplot as plt
import ip
import nn

checkpoint_path = "training/cp.ckpt"
(train_images, train_labels), (test_images,test_labels) = load_data()
model = create_model()
# history = model.fit(train_images, train_labels, epochs=15, batch_size=256, verbose=1)
model.load_weights(checkpoint_path)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print test_loss, test_acc
def predict(img):
	p=model.predict(np.array([img.flatten(),]))
	return p
# labels=['ka','kha','ga','gha','knya','cha','chha','ja','jha','kni',]

img = cv2.imread('a.jpg',cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img,(5,5),0)
ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
bin_img = np.zeros((img.shape),dtype='float32')

height,width = img.shape
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
		character = cv2.bitwise_not(character)
		resized_img = cv2.resize(character,(32,32))
		p=predict(resized_img)
		print(np.argmax(p))
# 		print(labels[np.argmax(p)])
		
