"""Ekspor bobot model LSTM dan parameter scaler ke format ringan (NumPy + JSON)
sehingga aplikasi web TIDAK memerlukan TensorFlow saat deploy.

Jalankan SEKALI di komputer lokal setelah notebook selesai (butuh TensorFlow):
    python export_for_deploy.py
"""
import json
import os

import joblib
import numpy as np
from tensorflow.keras.models import load_model

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

model = load_model(os.path.join(ART, "model_lstm.keras"), compile=False)
weights = model.get_weights()
expected = 10   # LSTM(3) + LSTM(3) + Dense(2) + Dense(2); layer Dropout tidak punya bobot
assert len(weights) == expected, f"Jumlah array bobot {len(weights)}, diharapkan {expected}. Arsitektur model berubah?"
np.savez_compressed(os.path.join(ART, "weights.npz"), *weights)

sx = joblib.load(os.path.join(ART, "scaler_X.joblib"))
sy = joblib.load(os.path.join(ART, "scaler_y.joblib"))
params = {
    "x_min": sx.data_min_.tolist(), "x_max": sx.data_max_.tolist(),
    "y_min": float(sy.data_min_[0]), "y_max": float(sy.data_max_[0]),
}
with open(os.path.join(ART, "scaler_params.json"), "w") as f:
    json.dump(params, f, indent=2)

# --- Verifikasi: hasil NumPy harus sama dengan Keras ---
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import numpy_predict  # noqa: E402

cfg = json.load(open(os.path.join(ART, "config.json")))
rng = np.random.default_rng(0)
x = rng.random((5, cfg["timesteps"], len(cfg["features"]))).astype("float32")
keras_out = model.predict(x, verbose=0).ravel()
numpy_out = np.array([numpy_predict(w, weights_list=[np.asarray(a) for a in weights]) for w in x])
print("Selisih maksimum Keras vs NumPy:", float(np.abs(keras_out - numpy_out).max()))
print("OK. File dibuat: artifacts/weights.npz dan artifacts/scaler_params.json")