import cv2
import numpy as np
# from matplotlib import pyplot as plt
import ip
import nn
import os

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
(train_images, train_labels), (test_images,test_labels) = load_data()
# Create checkpoint callback during initial training
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
#                                                  save_weights_only=True,
#                                                  verbose=1)
model = create_model()
# Train the model and save it to training/cp.ckpt during initial training
# model.fit(train_images, train_labels,  epochs = 15, 
#           validation_data = (test_images,test_labels),
#           callbacks = [cp_callback])
# history = model.fit(train_images, train_labels, epochs=15, batch_size=256, verbose=1)

# Load the model from training folder
model.load_weights(checkpoint_path)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print test_loss, test_acc
def predict(img):
	return model.predict(np.array([img.flatten(),]))

# labels=['ka','kha','ga','gha','knya','cha','chha','ja','jha','yna','taamatar','thaa','daa',''dhaa,'adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma','yaw','ra','la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya']

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
# 		print(np.argmax(p))
		print(labels[np.argmax(p)])
		
