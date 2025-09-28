import argparse, os, io, json, pandas as pd, numpy as np, joblib, tensorflow as tf
from urllib.request import urlopen
from pathlib import Path
import glob

FEATURES = [
    "hour_sin","hour_cos","dow","month","is_weekend",
    "Temp_prom_nac","Lluvia_total_nac","Viento_prom_nac",
    "lluvia_fuerte_share","viento_fuerte_share","ola_calor_share",
    "lluvia_sum_6h","viento_max_6h","temp_mean_6h",
    "llamadas_lag1","tmo_lag1","llamadas_lag24","tmo_lag24","llamadas_rolling_mean_3h"
]

# artefactos en la RAÍZ del repo (como los subiste)
MODEL_FILE   = "modelo_trafico_clima_v2.h5"
SCALER_X_PKL = "scaler_X_v2.pkl"
SCALER_Y_PKL = "scaler_y_v2.pkl"

DEFAULT_CSV_CANDIDATES = [
    "dataset_trafico_clima_nacional_v2.csv",
    "*trafico_clima*_v2*.csv",
]

def find_local_csv():
    for pattern in DEFAULT_CSV_CANDIDATES:
        for p in glob.glob(pattern):
            return p
    return None

def load_df(path_or_url: str):
    if path_or_url.startswith(("http://","https://")):
        with urlopen(path_or_url) as r: data = r.read()
        return pd.read_csv(io.BytesIO(data), parse_dates=["fecha_dt"])
    else:
        return pd.read_csv(path_or_url, parse_dates=["fecha_dt"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Ruta o URL al CSV con las 19 features")
    ap.add_argument("--out", default="public/predicciones.json", help="Archivo JSON de salida")
    args = ap.parse_args()

    for p in [MODEL_FILE, SCALER_X_PKL, SCALER_Y_PKL]:
        if not Path(p).exists():
            raise SystemExit(f"❌ Falta archivo requerido: {p}")

    csv_path = args.csv.strip() or find_local_csv()
    if not csv_path:
        raise SystemExit("❌ No encontré el CSV local. Sube dataset_trafico_clima_nacional_v2.csv o pasa --csv URL_PUBLICA")
    print(f"ℹ️ CSV: {csv_path}")

    model   = tf.keras.models.load_model(MODEL_FILE)
    scalerX = joblib.load(SCALER_X_PKL)
    scalerY = joblib.load(SCALER_Y_PKL)

    df = load_df(csv_path)
    miss = [c for c in FEATURES if c not in df.columns]
    if miss: raise SystemExit(f"❌ Faltan columnas en CSV: {miss}")

    X  = df[FEATURES].astype(float).values
    Xs = scalerX.transform(X)
    y_s = model.predict(Xs)
    y   = scalerY.inverse_transform(y_s)
    y[:, 1] = np.expm1(y[:, 1])  # revertir log de TMO

    items = []
    for i, row in df.iterrows():
        items.append({
            "fecha_dt": row["fecha_dt"].isoformat() if pd.notna(row["fecha_dt"]) else None,
            "hora": int(row["hora"]) if ("hora" in df.columns and pd.notna(row["hora"])) else None,
            "pred_llamadas": float(y[i, 0]),
            "pred_tmo_min":  float(y[i, 1]),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "source": csv_path,
            "count": len(items),
            "items": items[:500]
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Predicciones guardadas en {args.out} ({len(items)} filas)")

if __name__ == "__main__":
    main()
