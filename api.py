from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Request JSON kosong"}), 400

    dataset = data.get("dataset")
    training = data.get("training", {})

    if not dataset or len(dataset) < 10:
        return jsonify({"status": "error", "message": "Dataset minimal 10 baris"}), 400

    try:
        df = pd.DataFrame(dataset)

        # =========================
        # VALIDASI (TANPA AFKIR)
        # =========================
        required_cols = [
            "umur_ayam",
            "jumlah_ayam",
            "pakan_total_kg",
            "kematian",
            "telur_kg"
        ]

        for c in required_cols:
            if c not in df.columns:
                return jsonify({"status": "error", "message": f"Kolom '{c}' tidak ditemukan"}), 400

        # =========================
        # KONVERSI & CLEANING
        # =========================
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=required_cols, inplace=True)

        # =========================
        # FEATURE ENGINEERING
        # =========================
        df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
        df["persentase"] = (df["telur_kg"] / df["jumlah_ayam"]) * 100

        # HANDLE INF / NAN
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        X = df[[
            "umur_ayam",
            "jumlah_ayam",
            "pakan_per_ayam",
            "kematian",
            "persentase"
        ]]
        y = df["telur_kg"]

        # =========================
        # SPLIT DATA
        # =========================
        n_estimators = int(training.get("n_estimators", 150))
        random_state = int(training.get("random_state", 42))
        max_depth = training.get("max_depth", 6)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # =========================
        # TRAIN MODEL
        # =========================
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # =========================
        # EVALUASI
        # =========================
        MAE = mean_absolute_error(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(y_test, y_pred)

        avg_ayam = X_test["jumlah_ayam"].mean()
        MAE_per_ayam = MAE / avg_ayam
        MSE_per_ayam = MSE / (avg_ayam ** 2)
        RMSE_per_ayam = RMSE / avg_ayam

        # =========================
        # RINGKASAN PRODUKSI
        # =========================
        harian_telur_kg = y.mean()
        bulanan_telur_kg = y.sum()
        telur_per_ayam = harian_telur_kg / df["jumlah_ayam"].mean()

        # FIX KONVERSI
        harian_telur_butir = harian_telur_kg / 0.0625
        bulanan_telur_butir = bulanan_telur_kg / 0.0625

        # =========================
        # SAVE MODEL
        # =========================
        with open("model_telur.pkl", "wb") as f:
            pickle.dump(model, f)

        # =========================
        # OUTPUT
        # =========================
        return jsonify({
            "status": "success",
            "MAE_kg": round(MAE, 3),
            "MSE_kg": round(MSE, 3),
            "RMSE_kg": round(RMSE, 3),
            "MAE_per_ayam": round(MAE_per_ayam, 6),
            "MSE_per_ayam": round(MSE_per_ayam, 6),
            "RMSE_per_ayam": round(RMSE_per_ayam, 6),
            "R2": round(float(R2), 3),
            "Train_rows": len(X_train),
            "Test_rows": len(X_test),
            "Features_used": list(X.columns),
            "prediksi": {
                "harian_telur_kg": round(harian_telur_kg, 2),
                "bulanan_telur_kg": round(bulanan_telur_kg, 2),
                "telur_per_ayam": round(telur_per_ayam, 4),
                "harian_telur_butir": int(round(harian_telur_butir)),
                "bulanan_telur_butir": int(round(bulanan_telur_butir))
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    data = request.get_json()

    try:
        # =========================
        # LOAD MODEL
        # =========================
        try:
            with open("model_telur.pkl", "rb") as f:
                model = pickle.load(f)
        except:
            return jsonify({"status": "error", "message": "Model belum di-train"}), 400

        # =========================
        # INPUT DARI UI
        # =========================
        umur = float(data.get("umur", 0))
        jumlah_ayam = float(data.get("jumlah_ayam", 0))
        pakan = float(data.get("pakan_total_kg", 0))
        kematian = float(data.get("kematian", 0))
        persentase = float(data.get("persentase", 0))

        if jumlah_ayam <= 0:
            return jsonify({"status": "error", "message": "Jumlah ayam harus > 0"}), 400

        # =========================
        # FEATURE ENGINEERING
        # =========================
        pakan_per_ayam = pakan / jumlah_ayam

        X_input = [[
            umur,
            jumlah_ayam,
            pakan_per_ayam,
            kematian,
            persentase
        ]]

        # =========================
        # PREDIKSI MODEL
        # =========================
        pred_kg = float(model.predict(X_input)[0])
        pred_kg = max(pred_kg, 0)

        # =========================
        # KONVERSI
        # =========================
        telur_butir = pred_kg / 0.0625
        telur_per_ayam = pred_kg / jumlah_ayam

        # =========================
        # OUTPUT
        # =========================
        return jsonify({
            "status": "success",
            "prediksi": {
                "harian_telur_kg": round(pred_kg, 2),
                "bulanan_telur_kg": round(pred_kg * 30, 2),
                "telur_per_ayam": round(telur_per_ayam, 4),
                "harian_telur_butir": int(round(telur_butir)),
                "bulanan_telur_butir": int(round(telur_butir * 30))
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
