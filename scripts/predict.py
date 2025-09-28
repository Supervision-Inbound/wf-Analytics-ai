# scripts/predict.py — versión tolerante con diagnóstico claro
import argparse, os, io, json, glob
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# === columnas esperadas (19 features) ===
FEATURES = [
    "hour_sin","hour_cos","dow","month","is_weekend",
    "Temp_prom_nac","Lluvia_total_nac","Viento_prom_nac",
    "lluvia_fuerte_share","viento_fuerte_share","ola_calor_share",
    "lluvia_sum_6h","viento_max_6h","temp_mean_6h",
    "llamadas_lag1","tmo_lag1","llamadas_lag24","tmo_lag24","llamadas_rolling_mean_3h"
]

# artefactos (en la raíz del repo, como los subiste)
MODEL_FILE   = "modelo_trafico_clima_v2.h5"
SCALER_X_PKL = "scaler_X_v2.pkl"
SCALER_Y_PKL = "scaler_y_v2.pkl"

# candidatos de CSV local
LOCAL_CSV_CANDIDATES = [
    "dataset_trafico_clima_nacional_v2.csv",
    "*trafico*_clima*_v2*.csv",
]


def _read_csv(path_or_url: str) -> pd.DataFrame:
    """Lee CSV desde ruta o URL. NO obliga 'fecha_dt' ni 'hora'."""
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url) as r:
            data = r.read()
        df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_csv(path_or_url)

    # si existe 'fecha_dt', parsea a datetime (si no, la dejamos ausente)
    if "fecha_dt" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    # si existe 'hora', intenta a entero
    if "hora" in df.columns:
        df["hora"] = pd.to_numeric(df["hora"], errors="coerce").astype("Int64")

    return df


def _find_local_csv() -> str | None:
    for pat in LOCAL_CSV_CANDIDATES:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Ruta o URL del CSV con las 19 features")
    ap.add_argument("--out", default="public/predicciones.json", help="Salida JSON")
    args = ap.parse_args()

    # --- verificar artefactos ---
    missing_files = [p for p in [MODEL_FILE, SCALER_X_PKL, SCALER_Y_PKL] if not Path(p).exists()]
    if missing_files:
        raise SystemExit(f"❌ Faltan artefactos del modelo en la raíz: {missing_files}")

    # --- resolver CSV ---
    csv_path = args.csv.strip() or _find_local_csv()
    if not csv_path:
        raise SystemExit("❌ No encontré CSV local. Sube 'dataset_trafico_clima_nacional_v2.csv' o usa --csv URL")

    print(f"ℹ️ CSV a usar: {csv_path}")

    # --- leer CSV y mostrar diagnóstico ---
    df = _read_csv(csv_path)
    print("🔎 Columnas detectadas:", list(df.columns))
    print("🔎 Primeras filas:\n", df.head(3))

    # --- validar columnas necesarias ---
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"❌ Faltan columnas requeridas para el modelo: {missing_cols}\n"
                         f"✅ Presentes: {[c for c in FEATURES if c in df.columns]}")

    # --- cargar modelo y scalers ---
    print("⏳ Cargando modelo y scalers…")
    model   = tf.keras.models.load_model(MODEL_FILE)
    scalerX = joblib.load(SCALER_X_PKL)
    scalerY = joblib.load(SCALER_Y_PKL)

    # --- preparar y predecir ---
    X  = df[FEATURES].astype(float).values
    Xs = scalerX.transform(X)
    print("⏳ Inferencia…")
    y_pred_s = model.predict(Xs, verbose=0)
    y_pred   = scalerY.inverse_transform(y_pred_s)

    # índice 1 corresponde a 'tmo' → revertir log1p aplicado en entrenamiento
    if y_pred.shape[1] >= 2:
        y_pred[:, 1] = np.expm1(y_pred[:, 1])

    # --- armar JSON de salida (fecha/hora son opcionales) ---
    items = []
    has_fecha = "fecha_dt" in df.columns
    has_hora  = "hora" in df.columns
    for i in range(len(df)):
        item = {
            "pred_llamadas": float(y_pred[i, 0]),
            "pred_tmo_min":  float(y_pred[i, 1]) if y_pred.shape[1] >= 2 else None,
        }
        if has_fecha:
            val = df.at[i, "fecha_dt"]
            item["fecha_dt"] = (pd.to_datetime(val, errors="coerce").isoformat()
                                if pd.notna(val) else None)
        else:
            item["fecha_dt"] = None

        if has_hora:
            hv = df.at[i, "hora"]
            item["hora"] = int(hv) if pd.notna(hv) else None
        else:
            item["hora"] = None

        items.append(item)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": pd.Timestamp.utcnow().isoformat(),
                "source": csv_path,
                "count": len(items),
                "items": items[:500],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ Predicciones guardadas en {args.out} (filas: {len(items)})")


if __name__ == "__main__":
    main()
