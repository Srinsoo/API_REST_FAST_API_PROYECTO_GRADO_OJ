import pandas as pd
import requests
import re
import json
from xgboost import XGBRegressor
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_modelXGBoost(df):
    MODELS_DIR = Path("Models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print ("comenzando")
    #1 Validación inicial
    if df.empty:
        return "Error: Dataframe sin registros"
    df["Time"] = pd.to_datetime(df["Time"])

    df = df.sort_values("Time").reset_index(drop=True)
    df = df.set_index("Time")

    TARGET = "electricity_demand_values[kw]"

    # ==========================================
    # 2. LIMPIEZA METEOROLÓGICA
    # ==========================================

    weather_columns = [
        "air_pressure[mmHg]",
        "air_temperature[degree celcius]",
        "relative_humidity[%]",
        "wind_speed[M/S]",
        "solar_irridiation[W/m²]"
    ]

    for col in weather_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[weather_columns] = df[weather_columns].interpolate(
        method="linear",
        limit_direction="both"
    )

# Nubosidad
    df["total_cloud_cover[from ten]"] = (
        df["total_cloud_cover[from ten]"]
        .fillna("unknown")
        .astype(str)
    )

    df = pd.get_dummies(
        df,
        columns=["total_cloud_cover[from ten]"],
        dtype=int
    )

    # ==========================================
    # 3. VARIABLES TEMPORALES
    # ==========================================

    df["hora"] = df.index.hour
    df["dia_semana"] = df.index.dayofweek
    df["mes"] = df.index.month
    df["dia_mes"] = df.index.day

    df["es_fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    # Ciclos diarios
    df["hora_sin"] = np.sin(
        2 * np.pi * df["hora"] / 24
    )

    df["hora_cos"] = np.cos(
        2 * np.pi * df["hora"] / 24
    )

    # Ciclos semanales
    df["dia_sin"] = np.sin(
        2 * np.pi * df["dia_semana"] / 7
    )

    df["dia_cos"] = np.cos(
        2 * np.pi * df["dia_semana"] / 7
    )

    # ==========================================
    # 4. VARIABLES LAG
    # ==========================================

    for lag in [1, 2, 3, 24, 48, 72, 168]:
        df[f"consumo_lag_{lag}"] = (
            df[TARGET].shift(lag)
        )

    # Promedios móviles históricos
    for window in [3, 24, 168]:
        df[f"consumo_mean_{window}"] = (
            df[TARGET]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # Eliminar filas incompletas
    df = df.dropna()

    # ==========================================
    # 5. LIMPIAR NOMBRES DE COLUMNAS
    # ==========================================

    df.columns = [
        re.sub(
            r"_+",
            "_",
            re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
        ).strip("_")
        for col in df.columns
    ]

    TARGET = "electricity_demand_values_kw"

    # ==========================================
    # 6. DEFINIR VARIABLES
    # ==========================================

    features = [
        col for col in df.columns
        if col != TARGET
        and col != "heat_demand_values_kw"
    ]

    # Guardar las características utilizadas por el modelo
    with open(MODELS_DIR / "features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=4)

# Guardar las columnas correspondientes a las categorías de nubosidad
    cloud_columns = [
        col for col in features
        if col.startswith("total_cloud_cover_from_ten_")
    ]

    with open(MODELS_DIR / "cloud_columns.json", "w", encoding="utf-8") as f:
        json.dump(cloud_columns, f, indent=4)

    X = df[features]
    y = df[TARGET]

    print("Cantidad de variables:", len(features))
    print("Variables:", features)

    # ==========================================
    # 7. DIVISIÓN CRONOLÓGICA
    # ==========================================

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print("Entrenamiento:", len(X_train))
    print("Validación:", len(X_val))
    print("Prueba:", len(X_test))

    # ==========================================
    # 8. MODELO XGBOOST
    # ==========================================

    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100
    )

    # ==========================================
    # 9. ENTRENAMIENTO
    # ==========================================

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    print("Mejor iteración:", model.best_iteration)

    # ==========================================
    # 10. EVALUACIÓN
    # ==========================================

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    mask = np.abs(y_test) > 1e-6

    mape = (
        np.mean(
            np.abs(
                (y_test[mask] - predictions[mask]) /
                y_test[mask]
            )
        ) * 100
        if mask.any() else np.nan
    )

    wape = (
        np.sum(np.abs(y_test - predictions)) /
        np.sum(np.abs(y_test)) * 100
        if np.sum(np.abs(y_test)) > 0 else np.nan
    )

    metrics = {
        "MAE_kW": round(float(mae), 4),
        "RMSE_kW": round(float(rmse), 4),
        "MAPE_percent": round(float(mape), 2)
        if np.isfinite(mape) else None,
        "WAPE_percent": round(float(wape), 2)
        if np.isfinite(wape) else None,
        "R2": round(float(r2), 4)
    }

    print("\nRESULTADOS DEL MODELO")
    print(json.dumps(metrics, indent=4))


    fecha = datetime.now().strftime("%Y-%m-%d")
    model_path = MODELS_DIR / "Modelo_XGboost.json"
    model.save_model(str(model_path))

    

    return "Modelo entrenado exitosamente!"


