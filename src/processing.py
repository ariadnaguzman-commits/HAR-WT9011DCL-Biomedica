from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from src.config import SENSOR_COLS, WINDOW_SIZE, STEP_SIZE
except ImportError:
    from config import SENSOR_COLS, WINDOW_SIZE, STEP_SIZE

CHANNELS_8 = SENSOR_COLS + ["acc", "giro"]
STATS_6 = ["media", "desv", "min", "max", "rango", "rms"]
FEATURE_NAMES = [f"{ch}_{st}" for ch in CHANNELS_8 for st in STATS_6]


def cargar_archivo(filepath: str | Path) -> pd.DataFrame:
    """Carga un archivo .csv o .mat y estandariza las columnas del sensor."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".mat":
        mat = loadmat(path, squeeze_me=True)
        data_dict = {}
        for k, v in mat.items():
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 1:
                data_dict[k.lower()] = v
        if not data_dict:
            # Si viene como matriz única (N x 6) bajo alguna clave
            for k, v in mat.items():
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 6:
                    for idx, col in enumerate(SENSOR_COLS):
                        data_dict[col] = v[:, idx]
                    break
        df = pd.DataFrame(data_dict)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .csv o .mat")

    # Normalizar nombres de columnas a minúsculas y quitar espacios
    df.columns = [str(c).strip().lower() for c in df.columns]

    faltantes = [c for c in SENSOR_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en {path.name}: {faltantes}")

    # Asegurar tipos numéricos en señales del sensor
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)

    return df


def extraer_caracteristicas_ventana(ventana_6ejes: np.ndarray) -> np.ndarray:
    """
    Calcula las 48 características estadísticas para una ventana de forma (N, 6)
    con columnas ordenadas como [ax, ay, az, gx, gy, gz].
    """
    acc_mag = np.linalg.norm(ventana_6ejes[:, 0:3], axis=1, keepdims=True)
    giro_mag = np.linalg.norm(ventana_6ejes[:, 3:6], axis=1, keepdims=True)
    datos_8 = np.hstack([ventana_6ejes, acc_mag, giro_mag])  # (N, 8)

    media = np.mean(datos_8, axis=0)
    desv = np.std(datos_8, axis=0, ddof=1) if datos_8.shape[0] > 1 else np.zeros(8)
    minimo = np.min(datos_8, axis=0)
    maximo = np.max(datos_8, axis=0)
    rango = maximo - minimo
    rms = np.sqrt(np.mean(np.square(datos_8), axis=0))

    # Intercalar por canal: [ax_media, ax_desv, ..., giro_rms] -> 48 valores
    matriz_stats = np.vstack([media, desv, minimo, maximo, rango, rms])  # (6, 8)
    return matriz_stats.T.ravel()


def procesar_dataframe(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    discard_initial: int = 0,
) -> pd.DataFrame:
    """Segmenta un DataFrame en ventanas deslizantes y devuelve la tabla de 48 características."""
    if discard_initial > 0 and len(df) > discard_initial + window_size:
        df = df.iloc[discard_initial:].reset_index(drop=True)

    n_muestras = len(df)
    if n_muestras < window_size:
        raise ValueError(
            f"Señal demasiado corta ({n_muestras} muestras). Se requieren al menos {window_size}."
        )

    señal = df[SENSOR_COLS].to_numpy(dtype=np.float64)
    tiene_etiqueta = "movimiento" in df.columns

    filas_features = []
    etiquetas = []
    indices_inicio = []
    indices_fin = []

    for inicio in range(0, n_muestras - window_size + 1, step_size):
        fin = inicio + window_size
        ventana = señal[inicio:fin]
        feats = extraer_caracteristicas_ventana(ventana)
        filas_features.append(feats)
        indices_inicio.append(inicio)
        indices_fin.append(fin)

        if tiene_etiqueta:
            etiqueta_moda = df["movimiento"].iloc[inicio:fin].mode().iloc[0]
            etiquetas.append(str(etiqueta_moda).strip().lower())

    df_feats = pd.DataFrame(filas_features, columns=FEATURE_NAMES)
    df_feats.insert(0, "muestra_fin", indices_fin)
    df_feats.insert(0, "muestra_inicio", indices_inicio)

    if tiene_etiqueta:
        df_feats["movimiento"] = etiquetas

    return df_feats