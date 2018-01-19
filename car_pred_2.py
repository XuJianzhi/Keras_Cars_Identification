
#https://github.com/XuJianzhi/Keras_Cars_Identification/edit/master/car_pred.py
#XuJianzhi 2018.1.17
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input

import keras, cv2, os, numpy as np, pandas as pd, time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


x_all = []
y_all = []

path_in = '/home/m/桌面/cars/'
list_1 = ['jili','baoma','mumaren']
for car_name in list_1:
	list_2 = os.listdir(path_in + car_name)
	for img_name in tqdm(list_2[:400]):
		img = load_img(path_in + car_name + '/' + img_name, target_size=(64,64))
		#img = img_to_array(img).flatten() / 255
		img = img_to_array(img) / 255
		img = [img.tolist()]
		x_all += img
		y_all += [car_name]

#print(np.array(x_all).shape)
#print(np.array(y_all).shape)

	

x_train, x_test, y_train, y_test = train_test_split(np.array(x_all), np.array(y_all))
del x_all, y_all

classes = len(list_1)		# 几种车

encoder = LabelEncoder()
hotencoder = OneHotEncoder(sparse=False)
y_train = hotencoder.fit_transform(pd.DataFrame({'a':encoder.fit_transform(y_train)}))
y_test = hotencoder.transform(pd.DataFrame({'a':encoder.transform(y_test)}))


model = Sequential()
#model.add(Dense(input_dim=100*100,units=633,activation='relu'))
#model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))	#(100,100,3), data_format='channels_last'))		#input_dim=100*100)),
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
'''
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
'''
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dense(classes, activation='softmax', name='predictions'))

#model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])	

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=50)

result = model.evaluate(x_test, y_test)
print(result)






###########
path_in = '/home/m/桌面/cars/'
list_1 = ['jili','baoma','mumaren']
for car_name in list_1:
	list_2 = os.listdir(path_in + car_name)
	for img_name in tqdm(list_2[400:500]):
		img = load_img(path_in + car_name + '/' + img_name, target_size=(70,70))
		#img = img_to_array(img).flatten() / 255
		img = img_to_array(img) / 255
		img = [img.tolist()]
		x_all += img
		y_all += [car_name]

x_test = np.array(x_all)
y_test = np.array(y_all)

y_test = hotencoder.transform(pd.DataFrame({'a':encoder.transform(y_test)}))

y_pred = model.predict(x_test)
y_pred[y_pred<1.0] = 0

accu = 1 - (len(y_pred)*classes - sum((y_pred == y_test).flatten()))/(2*len(y_pred))







