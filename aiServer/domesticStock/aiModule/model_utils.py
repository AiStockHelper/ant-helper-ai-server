# model_utils.py
import pickle
from tensorflow.keras.models import load_model

def save_model(model, scaler, name="stock_model"):
    model.save(f"{name}.h5")
    with open(f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load_saved_model(name="stock_model"):
    model = load_model(f"{name}.h5")
    with open(f"{name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler