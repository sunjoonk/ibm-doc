import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization # Conv1D, Flatten 추가
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import roc_auc_score

# 1. 데이터
path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=8282, shuffle=True, stratify=y
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=5234, stratify=y_train
)

scalers = [
    ('None', None), ('MinMax', MinMaxScaler()), ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()), ('Robust', RobustScaler())
]

original_x_train = x_train.copy()
original_x_test = x_test.copy()
original_x_val = x_val.copy()
original_test_csv = test_csv.copy()

for scaler_name, scaler in scalers:
    print(f"\n\n======= 스케일러: {scaler_name} =======")
    x_train = original_x_train.copy().values
    x_test = original_x_test.copy().values
    x_val = original_x_val.copy().values
    test_csv_data = original_test_csv.copy().values

    if scaler is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)
        test_csv_data = scaler.transform(test_csv_data)
        
    # 요청사항 1: Conv1D 입력을 위한 데이터 Reshape
    x_train_reshaped = x_train.reshape(x_train.shape[0], 200, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 200, 1)
    x_val_reshaped = x_val.reshape(x_val.shape[0], 200, 1)
    test_csv_reshaped = test_csv_data.reshape(test_csv_data.shape[0], 200, 1)

    # 요청사항 2: 모델의 첫 번째 레이어를 Conv1D로 변경
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, input_shape=(200, 1), activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    es = EarlyStopping(monitor='val_auc', mode='max', patience=50, restore_best_weights=True)
    
    start_time = time.time()
    model.fit(
        x_train_reshaped, y_train, epochs=200, batch_size=128, verbose=1,
        validation_data=(x_val_reshaped, y_val), callbacks=[es]
    )
    end_time = time.time()

    # 4. 평가, 예측
    print("걸린 시간 :", round(end_time - start_time, 2), "초")
    results = model.evaluate(x_test_reshaped, y_test)
    y_pred = model.predict(x_test_reshaped)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"loss: {results[0]}, auc: {results[1]}")
    print(f"ROC AUC Score: {auc}")
