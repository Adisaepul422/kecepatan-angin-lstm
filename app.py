"""Aplikasi Flask untuk menampilkan data historis dan hasil prediksi model LSTM
(peramalan kecepatan angin per jam pada data SCADA turbin angin)."""
import json
import os
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")
# Aplikasi ini TIDAK memakai TensorFlow: bobot model diekspor ke NumPy (lihat export_for_deploy.py)
# sehingga ringan dan bisa dijalankan di Vercel/Render/Railway.
REQUIRED = ["config.json", "weights.npz", "scaler_params.json",
            "hourly_data.csv", "test_predictions.csv"]

app = Flask(__name__)
_state = {}
_lock = threading.Lock()


def artifacts_ready():
    return all(os.path.exists(os.path.join(ART, f)) for f in REQUIRED)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x, W, U, b, return_sequences):
    """Forward pass layer LSTM Keras (urutan gerbang: input, forget, cell, output)."""
    units = U.shape[0]
    h = np.zeros(units)
    c = np.zeros(units)
    outs = []
    for t in range(x.shape[0]):
        z = x[t] @ W + h @ U + b
        i = _sigmoid(z[:units])
        f = _sigmoid(z[units:2 * units])
        g = np.tanh(z[2 * units:3 * units])
        o = _sigmoid(z[3 * units:])
        c = f * c + i * g
        h = o * np.tanh(c)
        outs.append(h)
    return np.array(outs) if return_sequences else h


def numpy_predict(window_scaled, weights_list=None):
    """Prediksi satu sampel (timesteps, features) yang sudah di-scale -> nilai skala 0-1.
    Arsitektur: LSTM(64, seq) -> LSTM(32) -> Dense(16, relu) -> Dense(1). Dropout tidak aktif saat prediksi."""
    w = weights_list if weights_list is not None else load_artifacts()["weights"]
    h1 = _lstm(window_scaled.astype("float64"), w[0], w[1], w[2], True)
    h2 = _lstm(h1, w[3], w[4], w[5], False)
    d1 = np.maximum(h2 @ w[6] + w[7], 0.0)
    return float((d1 @ w[8] + w[9])[0])


def load_artifacts():
    """Muat bobot, parameter scaler, dan data sekali saja (aman untuk permintaan bersamaan)."""
    if _state.get("ready"):
        return _state
    with _lock:
        if _state.get("ready"):
            return _state
        loaded = {}
        with open(os.path.join(ART, "config.json")) as f:
            loaded["cfg"] = json.load(f)
        with open(os.path.join(ART, "scaler_params.json")) as f:
            loaded["scaler"] = json.load(f)
        npz = np.load(os.path.join(ART, "weights.npz"))
        loaded["weights"] = [npz[f"arr_{i}"].astype("float64") for i in range(len(npz.files))]
        loaded["hourly"] = (pd.read_csv(os.path.join(ART, "hourly_data.csv"), parse_dates=["datetime"])
                            .set_index("datetime").sort_index())
        loaded["test"] = pd.read_csv(os.path.join(ART, "test_predictions.csv"), parse_dates=["datetime"])
        loaded["ready"] = True
        _state.update(loaded)
    return _state


def fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


@app.route("/")
def index():
    return render_template("index.html", ready=artifacts_ready())


@app.route("/health")
def health():
    return jsonify(status="ok", artifacts_ready=artifacts_ready())


@app.route("/api/summary")
def api_summary():
    s = load_artifacts()
    cfg, h = s["cfg"], s["hourly"]
    return jsonify(
        timesteps=cfg["timesteps"], features=cfg["features"], max_value=cfg["max_value"], unit=cfg.get("unit", "m/s"),
        rmse=cfg["rmse"], mae=cfg["mae"],
        rmse_baseline=cfg["rmse_baseline"], mae_baseline=cfg["mae_baseline"],
        data_start=fmt(h.index[0]), data_end=fmt(h.index[-1]), n_hours=len(h),
    )


@app.route("/api/history")
def api_history():
    """Data historis. range = 7d | 30d | 90d | all (all dirata-ratakan harian)."""
    s = load_artifacts()
    target = s["cfg"]["target"]
    series = s["hourly"][target]
    rng = request.args.get("range", "30d")
    days = {"7d": 7, "30d": 30, "90d": 90}.get(rng)
    if days:
        series = series.iloc[-days * 24:]
    else:
        series = series.resample("1D").mean()
    return jsonify(labels=[fmt(t) for t in series.index],
                   values=[round(float(v), 2) for v in series.values], range=rng)


@app.route("/api/test")
def api_test():
    """Perbandingan nilai aktual vs prediksi pada data uji (n titik pertama)."""
    s = load_artifacts()
    n = min(int(request.args.get("n", 168)), 24 * 30)
    t = s["test"].iloc[:n]
    return jsonify(labels=[fmt(x) for x in t["datetime"]],
                   actual=[round(float(v), 2) for v in t["actual"]],
                   prediction=[round(float(v), 2) for v in t["prediction"]])


@app.route("/api/forecast")
def api_forecast():
    """Prediksi kecepatan angin satu jam setelah data terakhir, berdasarkan `timesteps` jam terakhir."""
    s = load_artifacts()
    cfg, h = s["cfg"], s["hourly"]
    sc = s["scaler"]
    window = h[cfg["features"]].iloc[-cfg["timesteps"]:].to_numpy(dtype="float64")
    x_min, x_max = np.array(sc["x_min"]), np.array(sc["x_max"])
    rng = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    window_scaled = (window - x_min) / rng
    pred_scaled = numpy_predict(window_scaled, s["weights"])
    value = pred_scaled * (sc["y_max"] - sc["y_min"]) + sc["y_min"]
    value = float(np.clip(value, 0, cfg["max_value"]))
    next_ts = h.index[-1] + pd.Timedelta(hours=1)
    return jsonify(forecast=round(value, 2), target_time=fmt(next_ts),
                   last_observed=round(float(h[cfg["target"]].iloc[-1]), 2),
                   last_observed_time=fmt(h.index[-1]))


@app.errorhandler(Exception)
def handle_error(e):
    if request.path.startswith("/api/"):
        return jsonify(error=str(e)), 500
    raise e


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)