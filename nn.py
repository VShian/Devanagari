import tensorflow as tf
from tensorflow import keras
import numpy as np

# data = np.genfromtxt('minidata.csv',delimiter=',')
# print(data[0])

def load_data():
  train_images, train_labels = [], []
  # test_images, test_labels = [], []
  f=open('data.csv','r')
  for line in f:
  	x_y = [int(x) for x in line.split(',')]
  	train_images.append(x_y[:-1])
  	train_labels.append(x_y[-1]-1)
  f.close()
  # f=open('minidata.csv','r')
  # for line in f:
  # 	x_y = [int(x) for x in line.split(',')]
  # 	test_images.append(x_y[:-1])
  # 	test_labels.append(x_y[-1]-1)
  # f.close()
  # train_labels=np.asarray(train_labels)
  # train_images=np.asarray(train_images)
  # test_images=np.asarray(test_images)
  # test_labels=np.asarray(test_labels)
  # print train_labels.shape,train_images.shape
  train=zip(train_images,train_labels)
  random.shuffle(train)
  train_images[:],train_labels[:]=zip(*train)

  test_images=np.array(train_images[-2000:])
  test_labels=np.array(train_labels[-2000:])
  train_images=np.array(train_images[:-2000])
  train_labels=np.array(train_labels[:-2000])
  
  return (train_images, train_labels), (test_images,test_labels)

def load_model(input_length):
    model = keras.Sequential([
      keras.layers.Dense(600, input_shape=(1024,), activation='relu'),
      keras.layers.Dense(800, activation='relu'),
      keras.layers.Dense(1000, activation='relu'),
      keras.layers.Dense(800, activation='relu'),
      keras.layers.Dense(600, activation='relu'),
      keras.layers.Dense(36, activation='softmax')
    ])

  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model

def train_model(train_images, train_labels, model):
  history = model.fit(train_images, train_labels, epochs=5, batch_size=256, verbose=1)
  return history,model

def test_model(test_images, test_labels, model):
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)
  return test_loss, test_acc


# predictions = model.predict(test_images)