# keras2/keras67_optimizer02_california.py

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
    random_state=1111,
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
        model.add(Dense(128, input_dim=8))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='linear')) 

        # 3. 컴파일, 훈련
        model.compile(loss='mse', optimizer=optim(learning_rate=lr))

        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        es = EarlyStopping( 
            monitor = 'val_loss',       
            mode = 'min',               
            patience=30,             
            restore_best_weights=True,  
        )
        start_time = time.time()
        hist = model.fit(x_train, y_train, 
                        epochs=100, 
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

        from sklearn.metrics import r2_score, mean_absolute_error
        try:
            r2_score = r2_score(y_test, results)
        except:
            r2_score = "Nan"
            
        # 최고값 갱신
        if r2_score > best_score:
            best_score = r2_score
            best_optim = optim.__name__
            best_lr = lr
        
        print(f'{optim.__name__},  {lr} 일때의 r2 스코어 :', r2_score)
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        # 반복문내에서 같은 print문을 2번이상쓰면 버퍼가 꼬여서 제대로 출력안되는 경우 있음
print("=======================================")
print(f'최고 r2 스코어 : {best_score:.4f}')
print(f'최적 optimizer : {best_optim}')
print(f'최적 learning_rate : {best_lr}')
print("=======================================")
