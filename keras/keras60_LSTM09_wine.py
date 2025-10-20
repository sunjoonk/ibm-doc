from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization  # LSTM 추가
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터 로드 및 전처리
datasets = load_wine()
x = datasets.data
y = datasets.target

# 원핫 인코딩
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=517, stratify=y
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
]

for scaler_name, scaler in scalers:
    # 데이터 스케일링
    if scaler is None:
        x_train_scaled = x_train
        x_test_scaled = x_test
    else:
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    
    # LSTM 입력을 위한 3D 변환 (samples, timesteps, features)
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 13, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 13, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델 구성 (LSTM 사용)
    model = Sequential()
    model.add(LSTM(64, input_shape=(13, 1), return_sequences=True))  # 첫 번째 LSTM 레이어
    model.add(Dropout(0.2))
    model.add(LSTM(32))  # 두 번째 LSTM 레이어
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))  # 다중 분류 출력층

    # 3. 컴파일 및 훈련
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )
    
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
        restore_best_weights=True
    )

    start_time = time.time()
    model.fit(
        x_train_reshaped, y_train,
        epochs=1000,
        batch_size=32,
        verbose=0,
        validation_split=0.2,
        callbacks=[es]
    )
    end_time = time.time()
    print('걸린시간:', round(end_time - start_time, 2), '초')
    
    # 4. 평가 및 예측
    results = model.evaluate(x_test_reshaped, y_test)
    print('loss:', results[0])
    print('(categorical)acc:', results[1])

    y_pred = model.predict(x_test_reshaped)
    y_pred = np.argmax(y_pred, axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_test_argmax, y_pred)
    f1 = f1_score(y_test_argmax, y_pred, average='macro')
    
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print('acc :', acc)
    print("F1-Score :", f1)
