# keras2/keras67_optimizer04_dacon_ddareung.py

import tensorflow as tf 
print(tf.__version__)   # 2.7.4
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

import sklearn as sk 
print(sk.__version__)
import tensorflow as tf 
print(tf.__version__)
import numpy as np

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
import pandas as pd

# 1. 데이터
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
    x, y,
    test_size=0.3,
    random_state=1111,
    stratify=y,
)

# 스케일러
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 파라미터튜닝
optimizers = [Adam, Adagrad, SGD, RMSprop]
learning_rates = [0.1, 0.01, 0.05, 0.001, 0.0001]
# 파라미터튜닝에서 가장 성능차이가 커지는 파라미터는 learning_rate

# 출력용 파라미터
best_score = -float('inf')
best_optim = None
best_lr = None

# 2. 모델구성
for optim in optimizers:
    for lr in learning_rates:
        model = Sequential()
        model.add(Dense(128, input_dim=8, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid')) 

        # 3. 컴파일, 훈련
        model.compile(loss='binary_crossentropy', optimizer=optim(learning_rate=lr))

        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        es = EarlyStopping( 
            monitor = 'val_loss',       
            mode = 'min',               
            patience=30,             
            restore_best_weights=True,  
        )
        start_time = time.time()
        hist = model.fit(x_train, y_train, 
                        epochs=200, 
                        batch_size=32, 
                        verbose=0, 
                        validation_split=0.2,
                        callbacks=[es],
                        ) 
        end_time = time.time()

        #################### 속도 측정 ####################
        if gpus:
            print('GPU 있다~')
        else:
            print('GPU 없다~')
        print("걸린 시간 :", round(end_time-start_time, 2), "초")
        #################### 속도 측정 ####################
            
        # 4. 평가, 예측
        loss = model.evaluate(x_test, y_test)
        results = model.predict(x_test)

        from sklearn.metrics import r2_score, accuracy_score
        try:
            acc = accuracy_score(y_test, results)
        except:
            acc = "Nan"
            
        # 최고값 갱신
        if acc > best_score:
            best_score = acc
            best_optim = optim.__name__
            best_lr = lr
        
        print(f'{optim.__name__},  {lr} 일때의 acc 스코어 :', acc)
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        
print("=======================================")
print(f'최고 r2 스코어 : {best_score:.4f}')
print(f'최적 optimizer : {best_optim}')
print(f'최적 learning_rate : {best_lr}')
print("=======================================")
