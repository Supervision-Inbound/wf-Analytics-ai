# scripts/predict.py — diagnóstico detallado + scalerX aplicado + carga sin recompilar
import argparse, os, io, json, glob, sys, traceback
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

# === artefactos esperados en la RAÍZ del repo ===
MODEL_FILE   = "modelo_trafico_clima_v2.h5"
SCALER_X_PKL = "scaler_X_v2.pkl"
SCALER_Y_PKL = "scaler_y_v2.pkl"

# === columnas esperadas (19 features) ===
FEATURES = [
    "hour_sin","hour_cos","dow","month","is_weekend",
    "Temp_prom_nac","Lluvia_total_nac","Viento_prom_nac",
    "lluvia_fuerte_share","viento_fuerte_share","ola_calor_share",
    "lluvia_sum_6h","viento_max_6h","temp_mean_6h",
    "llamadas_lag1","tmo_lag1","llamadas_lag24","tmo_lag24","llamadas_rolling_mean_3h"
]

# Si tu CSV usa nombres distintos, puedes mapearlos aquí:
RENOMBRES = {
    # "Temp_prom_pais": "Temp_prom_nac",
    # "Lluvia_total_pais": "Lluvia_total_nac",
}

LOCAL_CSV_CANDIDATES = [
    "dataset_trafico_clima_nacional_v2.csv",
    "*trafico*_clima*_v2*.csv",
]

def _size(path: str):
    try:
        return os.path.getsize(path)
    except Exception:
        return None

def _read_csv(path_or_url: str) -> pd.DataFrame:
    print(f"📥 Leyendo CSV desde: {path_or_url}")
    try_encodings = ["utf-8", "utf-8-sig", "latin-1"]
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url) as r:
            data = r.read()
        last_err = None
        for enc in try_encodings:
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception as e:
                last_err = e
        raise last_err
    else:
        last_err = None
        for enc in try_encodings:
            try:
                return pd.read_csv(path_or_url, encoding=enc)
            except Exception as e:
                last_err = e
        raise last_err

def _find_local_csv() -> str | None:
    for pat in LOCAL_CSV_CANDIDATES:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None

def _die(msg: str, ex: Exception | None = None):
    print("\n❌ " + msg)
    if ex is not None:
        print("—— Traceback ——")
        traceback.print_exc()
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Ruta o URL del CSV con las 19 features")
    ap.add_argument("--out", default="public/predicciones.json", help="Salida JSON")
    args = ap.parse_args()

    print("📂 Archivos en raíz:", os.listdir("."))
    print(f"ℹ️ Tamaños: {MODEL_FILE}={_size(MODEL_FILE)}, {SCALER_X_PKL}={_size(SCALER_X_PKL)}, {SCALER_Y_PKL}={_size(SCALER_Y_PKL)}")

    # artefactos presentes
    missing = [p for p in [MODEL_FILE, SCALER_X_PKL, SCALER_Y_PKL] if not Path(p).exists()]
    if missing:
        _die(f"Faltan artefactos del modelo en la raíz: {missing}")

    # CSV a utilizar
    csv_path = args.csv.strip() or _find_local_csv()
    if not csv_path:
        _die("No encontré CSV local. Sube 'dataset_trafico_clima_nacional_v2.csv' o usa --csv URL")
    print(f"ℹ️ CSV elegido: {csv_path}")

    # lectura CSV
    try:
        df = _read_csv(csv_path)
    except Exception as e:
        _die(f"No pude leer el CSV ({csv_path}). Exporta a CSV simple (coma) y UTF-8.", e)

    # renombres si corresponde
    if RENOMBRES:
        df = df.rename(columns=RENOMBRES)

    # normalizaciones “suaves”
    if "fecha_dt" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    if "hora" in df.columns:
        df["hora"] = pd.to_numeric(df["hora"], errors="coerce").astype("Int64")

    print("🔎 Columnas detectadas:", list(df.columns))
    print("🔎 Primeras filas:\n", df.head(3))

    # validar features
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        present = [c for c in FEATURES if c in df.columns]
        _die(f"FALTAN columnas requeridas para el modelo: {missing_cols}\nPRESENTES: {present}")

    # cargar modelo y scalers (SIN recompilar)
    print("⏳ Cargando modelo y scalers…")
    try:
        import joblib, tensorflow as tf
        # ← cambio clave: compile=False, evita problemas de versiones Keras/TF
        model   = tf.keras.models.load_model(MODEL_FILE, compile=False)
        scalerX = joblib.load(SCALER_X_PKL)
        scalerY = joblib.load(SCALER_Y_PKL)
        print("✅ Modelo y scalers cargados.")
    except Exception as e:
        _die("Error cargando modelo o scalers", e)

    # preparar e inferir (aplicando scalerX como en entrenamiento)
    try:
        from sklearn.utils.validation import check_array
        X  = df[FEATURES].astype(float).values
        check_array(X)
        Xs = scalerX.transform(X)

        print("⏳ Inferencia con X escalado…")
        y_pred_s = model.predict(Xs, verbose=0)

        # devolver y a escala original
        y_pred = scalerY.inverse_transform(y_pred_s)
        if y_pred.shape[1] >= 2:
            y_pred[:, 1] = np.expm1(y_pred[:, 1])  # revertir log para TMO
    except Exception as e:
        _die("Error durante la inferencia (forma de X, scalerX/scalerY o modelo).", e)

    # generar salida JSON
    items = []
    has_fecha = "fecha_dt" in df.columns
    has_hora  = "hora" in df.columns
    for i in range(len(df)):
        item = {
            "pred_llamadas": float(y_pred[i, 0]),
            "pred_tmo_min":  float(y_pred[i, 1]) if y_pred.shape[1] >= 2 else None,
            "fecha_dt": (df.at[i, "fecha_dt"].isoformat() if has_fecha and pd.notna(df.at[i, "fecha_dt"]) else None),
            "hora": (int(df.at[i, "hora"]) if has_hora and pd.notna(df.at[i, "hora"]) else None),
        }
        items.append(item)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": pd.Timestamp.utcnow().isoformat(),
             "source": csv_path, "count": len(items), "items": items[:500]},
            f, ensure_ascii=False, indent=2)

    print(f"✅ Predicciones guardadas en {args.out} (filas: {len(items)})")

if __name__ == "__main__":
    main()


