### <<10>>

# 데이터 출처
# https://dacon.io/competitions/open/236068/overview/description


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

# 1.데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#  shape 확인
print(train_csv.shape)          # (652, 9)
print(test_csv.shape)           # (116, 8)
print(submission_csv.shape)     # (116, 2)

# 컬럼확인
print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
print(test_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age'],
print(submission_csv.columns)
# Index(['ID', 'Outcome'],

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())   # 결측치 없음

#train_csv = train_csv.dropna()

###### x와 y 분리 ####
x = train_csv.drop(['Outcome'], axis=1) # (652, 9)
y = train_csv['Outcome']                # (652,)
print("ㅡㅡㅡㅡㅡㅡㅡ")
print(y.shape) 

# 결측치 처리 
# 특정 생물학적 데이터는 0이 될 수없음. 이 train 데이터는 결측치를 0으로 세팅해놔서 0을 nan으로 대체하고 결측치처리해야함
# 여기서 결측치 처리하는 이유는 Outcome(이진분류정답컬럼)에 있는 0을 nan처리하면 안되기때문
# 여기서 결측치 처리할때 dropna를 쓰면 안되는 이유 : 여기서 dropna를 하면 정답지(y)랑 행 갯수가 달라지고 학습-정답 매칭이 안되어서 제대로 학습을 할 수 없다.
x = x.replace(0, np.nan)    
#x = x.fillna(x.mean())
x = x.fillna(x.median())

# 데이터 불균형 확인
print(pd.value_counts(y))
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
print(np.unique(y, return_counts=True))     # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=69758282,
    shuffle=True,
)
print(x_train.shape, x_test.shape)  # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)  # (456,) (196,)

# 2.모델구성
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'],
              ) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping( 
    monitor = 'val_loss',            
    mode = 'min',               
    patience=115,             
    restore_best_weights=True,  
)
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=2,
                 verbose=3, 
                 validation_split=0.15,
                 callbacks=[es],
                 )
end_time = time.time()

print("=============== hist =================")
print(hist)
print("=============== hist.history =================")
print(hist.history) # loss, val_loss, acc, val_acc
print("=============== loss =================")
print(hist.history['loss'])
print("=============== val_loss =================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')   # plot (x, y, color= ....) : y값만 넣으면 x는 1부터 시작하는 정수 리스트
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('diabetes loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 유측 상단에 라벨표시 
plt.grid()  #격자표시
plt.show()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)  # [0.5852866768836975, 0.6632652878761292]
print("loss : ", results[0]) 
print("acc : ", results[1])  
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
y_pred = np.round(y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy_score)  

# submission.csv 생성
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['Outcome'] = y_submit
from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path}submission_{current_time}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)