# keras24_softmax2_wine_stratify.py

from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True)) 
#(array([0, 1, 2]), array([59, 71, 48], dtype=int64)

# print(x)
# [[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
# ...
#  [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
# print(y)    # 다중분류 - 원핫인코딩 필요
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# y 원핫인코딩
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)   # nparray로 반환시키는 객체
y = ohe.fit_transform(y)
#print(y)
print(y.shape)  #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 517,
    stratify=y, # y(타겟데이터)를 계층화하겠다. y의 데이터가 적거나 불균형할때 사용. y, y_train, y_test의 각 클래스의 비율을 동등하게한다.
)
print(np.unique(y_train, return_counts=True))   # (array([0., 1.]), array([284, 142], dtype=int64)) : 원핫 인코딩을 한 상태라 0, 1만 있음
print(np.unique(y_test, return_counts=True))    # (array([0., 1.]), array([72, 36], dtype=int64)) : 원핫 인코딩을 한 상태라 0, 1만 있음

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)

start_time=time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es],
)
end_time=time.time()
print('걸린시간:', end_time - start_time)


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('(categorical)acc:', results[1])

y_pred = model.predict(x_test)
# print(y_pred)
# [[0.32754213 0.34670812 0.32574973]
#  ...
#  [0.32754213 0.34670812 0.32574973]]
# print(y_test)
# [[0. 0. 1.]
# ...
#  [1. 0. 0.]]

y_pred = np.argmax(y_pred, axis=1)
print(y_pred)   # [1 1 1 0 2 2 0 2 1 0 1 2 2 2 1 1 0 1 1 0 0 0 1 2 2 1 0 0 1 1 2 1 0 2 2 0]
y_test = np.argmax(y_test, axis=1)
print(y_test)   # [1 1 0 0 2 2 0 2 1 0 1 2 2 1 1 1 0 1 1 0 0 0 1 2 2 1 0 0 1 1 2 1 0 2 2 0]

# acc
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)


y_pred_f1 = np.argmax(model.predict(x_test), axis=1)
print(y_pred_f1)    
# [1 1 1 0 2 2 0 2 1 0 1 2 2 2 1 1 0 1 1 0 0 0 1 2 2 1 0 0 1 1 2 1 0 2 2 0]

# y_pred_f1 = (y_pred_f1 > 0.5).astype(int) # 이렇게 하면 안되는이유 : y_pred_f1의 2 클래스 정보가 소실된다. (0.5 이상인 1,2는 같은 걸로 취급되버린다.)

# -> 멀티클래스 처리 : average 파라미터 추가 필요
# f1_score  (y가 이진데이터가 아닐때 average 파라미터를 넣어줘야한다.(세 개 이상의 클래스 중요도를 어떻게 배분해야할지 명시가 필요하기때문에))
f1 = f1_score(y_test, y_pred_f1, average='macro')   # macro : 동등분할
print("F1-Score :", f1)

# acc : 1.0
# F1-Score : 1.0
