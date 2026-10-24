# train_model.py
import tensorflow as tf
from keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 플리커넥티드레이어
# 복잡하게?
# 로스?
# 트랜스포터

# def create_model(input_shape):
#     model = Sequential([
#         LSTM(128, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(64),
#         Dropout(0.2),
#         Dense(32, activation="relu"),
#         Dense(1)  # 내일 종가 예측
#     ])
#     model.compile(optimizer="adam", loss="mse")
#     return model

def create_model(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(X, y):
    model = create_model(X.shape[1:])
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)
    return model