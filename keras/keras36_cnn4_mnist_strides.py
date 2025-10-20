# keras36_cnn4_mnist_strides.py

import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

#  x reshape -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(10, 10, 1)))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(Conv2D(16, (3,3)))
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 64)          640    = (3*3*1+1)*64

#  conv2d_1 (Conv2D)           (None, 6, 6, 32)          18464  = (3*3*64+1)*32

#  conv2d_2 (Conv2D)           (None, 4, 4, 16)          4624   = (3*3*32+1)*16

#  flatten (Flatten)           (None, 256 = (4*4*16))     0

#  dense (Dense)               (None, 16)                4112   = (256*16)+16

#  dense_1 (Dense)             (None, 16)                272    = (16*16)+16

#  dense_2 (Dense)             (None, 10)                170    = (16*10)+10

# =================================================================
# Total params: 28,282
# Trainable params: 28,282
# Non-trainable params: 0
# _________________________________________________________________

model = Sequential()
model.add(Conv2D(5, (2,2), strides=1, input_shape=(10, 10, 3)))   # stride(디폴트값 1) : kernel이 한번에 움직이는 보폭. 너무 늘리면 데이터소실될 수있다. 너무 줄여도 연산량이 너무 많다.
model.add(Conv2D(filters=4, kernel_size=(2,2)))                   
# model.add(Conv2D(3, (3,3)))                                     # 특징맵이 커널보다 작아지면 에러가 난다.
model.add(Flatten())
model.add(Dense(units=10))
# model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()
# (Conv2D(5, (3,3), strides=2, input_shape=(10, 10, 3)))
# _________________________________________________________________
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d_3 (Conv2D)           (None, 4, 4, 5)           140

#  conv2d_4 (Conv2D)           (None, 3, 3, 4)           84

#  flatten_1 (Flatten)         (None, 36)                0

#  dense_3 (Dense)             (None, 10)                370

#  dense_4 (Dense)             (None, 10)                110

# =================================================================
# Total params: 704
# Trainable params: 704
# Non-trainable params: 0
# _________________________________________________________________

# model.add(Conv2D(5, (2,2), strides=1, input_shape=(10, 10, 3)))
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d_3 (Conv2D)           (None, 9, 9, 5)           65

#  conv2d_4 (Conv2D)           (None, 8, 8, 4)           84

#  flatten_1 (Flatten)         (None, 256)               0

#  dense_3 (Dense)             (None, 10)                2570

#  dense_4 (Dense)             (None, 10)                110

# =================================================================
# Total params: 2,829
# Trainable params: 2,829
# Non-trainable params: 0
# _________________________________________________________________