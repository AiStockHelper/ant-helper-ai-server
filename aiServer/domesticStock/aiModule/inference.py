import numpy as np
from .data_loader import load_ohlcv_1year
from .feature_engineering import add_indicators
from .model_utils import load_saved_model

def predict_tomorrow(ticker: str):
    model, scaler = load_saved_model()

    df = load_ohlcv_1year(ticker)
    df = df.rename(columns={
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "종가": "Close",
        "거래량": "Volume",
        "등락률": "Change"
    })
    df = add_indicators(df)

    features = df.values
    scaled = scaler.transform(features)

    scaled = scaler.transform(df.values)

    if len(scaled) < 252:
        pad_len = 252 - len(scaled)
        padding = np.zeros((pad_len, scaled.shape[1]))
        window = np.vstack([padding, scaled])
    else:
        window = scaled[-252:]

    window = np.expand_dims(window, axis=0)

    pred_scaled = model.predict(window)[0][0]

    # 스케일링 역변환
    close_idx = list(df.columns).index("Close")
    dummy = np.zeros((1, len(df.columns)))
    dummy[0][close_idx] = pred_scaled
    unscaled = scaler.inverse_transform(dummy)[0][close_idx]

    return float(unscaled)