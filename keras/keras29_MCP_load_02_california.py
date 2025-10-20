# keras29_MCP_load_02_california.py

# (1) SSL 적용 (한번만 실행하면됨)
import ssl
import certifi
# Python이 certifi의 CA 번들을 기본으로 사용하도록 설정
# 이 코드는 Python 3.6 이상에서 잘 작동하는 경향이 있습니다.
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl.SSLContext.set_default_verify_paths = lambda self, cafile=None, capath=None, cadata=None: self.load_verify_locations(cafile=certifi.where())

# # (2) SSL 비활성화 (한번만 실행하면됨)
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk 
print(sk.__version__)
import tensorflow as tf 
print(tf.__version__)
import numpy as np

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#[실습]

# 1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x)    # 데이터가 실수로 되어있다. -> 회귀모델사용 (0,1 같은 데이터 : 분류모델사용)
print(y)    # 데이터가 실수로 되어있다. -> 회귀모델사용 (0,1 같은 데이터 : 분류모델사용)
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1111
)

# # 2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_dim=8))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear')) 

path = './_save/keras28_mcp/02_california/'
model = load_model(path + 'k28_0604_1155_0162-0.4374.hdf5')      # restore_best_weights=True일때 저장한 모델과 가중치 동일
# r2 스코어 : 0.6711345467593834

#model = load_model(path + 'keras27_3_save_model.h5')

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=30,             
    restore_best_weights=True,  
)

hist = model.fit(x_train, y_train, 
                 epochs=300, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es, mcp],
                 ) 
"""        

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)
