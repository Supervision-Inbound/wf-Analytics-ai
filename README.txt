Modelo tráfico+clima (Chile) v2 — ENTRENADO EN KAGGLE (sin Hugging Face)
Archivos:
- modelo_trafico_clima_v2.h5
- scaler_X_v2.pkl
- scaler_y_v2.pkl
- dataset_trafico_clima_nacional_v2.csv
- pred_calls.png, pred_tmo.png, metrics.json

Uso básico (inferencia local):
  import joblib, tensorflow as tf
  model = tf.keras.models.load_model('modelo_trafico_clima_v2.h5')
  scaler_X = joblib.load('scaler_X_v2.pkl')
  # Prepara X con las 19 features y aplica scaler_X.transform(X) antes de predecir.
