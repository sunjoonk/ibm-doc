# keras14_verbose.py

# 08-1 복사

import numpy as np 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense 

# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10]).T  # 벡터는 transpose를 해도 변화가 없다.
# y = np.array([1,2,3,4,5,6,7,8,9,10]).T  # 벡터는 trnaspose를 해도 변화가 없다.

# print(x)
# print(y)

# print(x.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산
# print(y.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산

# #exit()
# 훈련데이터(70%)
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

# 평가데이터(30%)
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))    # input 차원이 1(x컬럼 1)이면서 출력 1개(y컬럼 1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1,
          verbose=0,
          )

# 훈련과정을 로그로 보여주는 건 실제 훈련속도와 로그를 프린트해서 보여주는 속도의 괴리감으로 인해 훈련속도에 영향을 준다. 따라서 옵션으로 켜고 끌수있다.
# verbose 옵션으로 로그를 끄면 훈련 한번당 프린트를 할 필요가 없기때문에 학습속도를 더빨리 할 수있다.
# 0 : 침묵
# 1 : 디폴트
# 2 : 프로그레스바 삭제
# 3 : 에포만 나옴

# 4. 평가, 예측
loss =model.evaluate(x_test, y_test)
results = model.predict([11])   # 벡터를 predict하지만 케라스가 행렬 형태로 변환하여 계산한다.(2차원 출력)
print('loss :', loss)
print('[11]의 예측값 :', results)