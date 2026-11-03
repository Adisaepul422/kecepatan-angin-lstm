"""Aplikasi Flask untuk menampilkan data historis dan hasil prediksi model LSTM
(peramalan kecepatan angin per jam pada data SCADA turbin angin)."""
import json
import os
import threading

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")
REQUIRED = ["config.json", "model_lstm.keras", "scaler_X.joblib", "scaler_y.joblib",
            "hourly_data.csv", "test_predictions.csv"]

app = Flask(__name__)
_state = {}
_lock = threading.Lock()


def artifacts_ready():
    return all(os.path.exists(os.path.join(ART, f)) for f in REQUIRED)


def load_artifacts():
    """Muat model, scaler, dan data sekali saja (aman untuk permintaan bersamaan)."""
    if _state.get("ready"):
        return _state
    with _lock:                      # hanya satu permintaan yang memuat; yang lain menunggu
        if _state.get("ready"):
            return _state
        from tensorflow.keras.models import load_model  # import berat

        loaded = {}
        with open(os.path.join(ART, "config.json")) as f:
            loaded["cfg"] = json.load(f)
        loaded["model"] = load_model(os.path.join(ART, "model_lstm.keras"), compile=False)
        loaded["scaler_X"] = joblib.load(os.path.join(ART, "scaler_X.joblib"))
        loaded["scaler_y"] = joblib.load(os.path.join(ART, "scaler_y.joblib"))
        loaded["hourly"] = (pd.read_csv(os.path.join(ART, "hourly_data.csv"), parse_dates=["datetime"])
                            .set_index("datetime").sort_index())
        loaded["test"] = pd.read_csv(os.path.join(ART, "test_predictions.csv"), parse_dates=["datetime"])
        loaded["ready"] = True       # ditandai siap HANYA setelah semuanya selesai dimuat
        _state.update(loaded)
    return _state


def warm_up():
    """Muat model di latar belakang saat server start agar permintaan pertama tidak lama."""
    if artifacts_ready():
        threading.Thread(target=load_artifacts, daemon=True).start()


warm_up()


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
    window = h[cfg["features"]].iloc[-cfg["timesteps"]:]
    x = s["scaler_X"].transform(window).reshape(1, cfg["timesteps"], len(cfg["features"]))
    pred_scaled = s["model"].predict(x, verbose=0)
    value = float(s["scaler_y"].inverse_transform(pred_scaled)[0, 0])
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
