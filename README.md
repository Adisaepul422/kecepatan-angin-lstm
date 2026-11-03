# TA-04: Peramalan Kecepatan Angin per Jam Menggunakan LSTM pada Data SCADA Turbin Angin

Tugas Akhir 04 - Pembelajaran Mesin (Modul LSTM).
Model LSTM meramalkan **kecepatan angin satu jam ke depan** dari data 24 jam terakhir (kecepatan dan arah angin).
Dataset: [Wind Turbine SCADA Dataset (Kaggle)](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset).

## Isi folder

```
TA04_LSTM_Wind_Turbine/
├── TA_04_LSTM_Wind_Turbine.ipynb   # EDA, preprocessing, training, evaluasi, simpan model
├── app.py                          # aplikasi web Flask
├── templates/index.html            # tampilan dashboard
├── artifacts/                      # DIISI oleh notebook (model, scaler, config, data)
├── dokumen/                        # laporan TP-04
├── requirements.txt                # dependensi aplikasi
├── Procfile                        # perintah start untuk Railway/Render
├── .python-version                 # versi Python untuk deployment
└── T1.csv                          # <-- TARUH dataset di sini (tidak disertakan)
```

## Langkah 1: Persiapan lingkungan

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
pip install matplotlib ipykernel   # tambahan untuk menjalankan notebook
```

## Langkah 2: Latih model (menghasilkan folder `artifacts/`)

1. Salin `T1.csv` ke folder ini (sejajar dengan notebook).
2. Buka `TA_04_LSTM_Wind_Turbine.ipynb` di VS Code, pilih kernel `venv`, lalu **Run All**.
3. Setelah selesai, `artifacts/` berisi: `model_lstm.keras`, `model_lstm.h5`, `scaler_X.joblib`,
   `scaler_y.joblib`, `config.json`, `hourly_data.csv`, `test_predictions.csv`.
4. Lengkapi sel markdown berlabel "isi dengan temuanmu sendiri" dengan analisismu.

## Langkah 3: Jalankan aplikasi web secara lokal

```bash
python app.py
```

Buka http://localhost:5000. Endpoint API: `/api/summary`, `/api/history?range=30d`,
`/api/test?n=168`, `/api/forecast`, `/health`.

## Langkah 4: Upload ke GitHub

```bash
git init
git add .
git commit -m "TA-04 LSTM turbin angin"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

Pastikan folder `artifacts/` ikut ter-upload (dibutuhkan aplikasi). Ukuran file model kecil sehingga aman untuk GitHub.

## Langkah 5: Deploy (Render atau Railway)

**Render:** New > Web Service > hubungkan repo GitHub.
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --workers 1 --threads 2 --timeout 180`

**Railway:** New Project > Deploy from GitHub repo. Railway membaca `Procfile` otomatis.

**Catatan penting:**
- TensorFlow memakan memori besar. Paket gratis Render (512 MB) bisa kehabisan memori. Jika aplikasi crash saat memuat model,
  coba Railway atau naikkan paket instance.
- Versi TensorFlow di `requirements.txt` sebaiknya sama dengan versi lokal (`pip show tensorflow`),
  supaya model `.keras` bisa dimuat tanpa error.
- Request pertama bisa lambat karena model dimuat saat pertama kali diakses.

## Checklist pengumpulan TA-04

- [ ] Persetujuan dosen untuk studi kasus dan dataset
- [ ] Repositori GitHub (notebook, kode Flask, model, scaler)
- [ ] URL aplikasi yang sudah di-hosting
- [ ] Laporan lengkap (DOC dan PDF): justifikasi studi kasus, analisis data, arsitektur model, evaluasi, interpretasi
- [ ] Video presentasi YouTube (5-7 menit)
# kecepatan-angin-lstm
