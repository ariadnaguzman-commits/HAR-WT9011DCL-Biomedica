from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.config import (
        CLASES,
        DATA_RAW_DIR,
        DISCARD_INITIAL_SAMPLES,
        FIGURES_DIR,
        FS,
        MODEL_PATH,
        MODELS_DIR,
        RESULTS_DIR,
        STEP_SIZE,
        WINDOW_SIZE,
    )
    from src.processing import FEATURE_NAMES, cargar_archivo, procesar_dataframe
except ImportError:
    from config import (
        CLASES,
        DATA_RAW_DIR,
        DISCARD_INITIAL_SAMPLES,
        FIGURES_DIR,
        FS,
        MODEL_PATH,
        MODELS_DIR,
        RESULTS_DIR,
        STEP_SIZE,
        WINDOW_SIZE,
    )
    from processing import FEATURE_NAMES, cargar_archivo, procesar_dataframe


def inferir_clase_desde_nombre(stem: str) -> str:
    """Extrae la clase de movimiento desde el nombre del archivo si no viene como columna."""
    nombre = stem.lower()
    for clase in sorted(CLASES, key=len, reverse=True):
        if clase in nombre:
            return clase
    raise ValueError(f"No se pudo identificar la clase de movimiento en el archivo: {stem}")


def construir_dataset_ventanas(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Lee todos los .csv en data/raw/, segmenta en ventanas y extrae las 48 características."""
    archivos = sorted(raw_dir.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .csv en {raw_dir}")

    bloques = []
    for archivo in archivos:
        sujeto = archivo.stem.split("_")[0].upper()
        df_raw = cargar_archivo(archivo)

        if "movimiento" not in df_raw.columns:
            df_raw["movimiento"] = inferir_clase_desde_nombre(archivo.stem)

        df_ventanas = procesar_dataframe(
            df_raw,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            discard_initial=DISCARD_INITIAL_SAMPLES,
        )
        df_ventanas["sujeto"] = sujeto
        df_ventanas["archivo"] = archivo.name
        bloques.append(df_ventanas)

    dataset = pd.concat(bloques, ignore_index=True)
    dataset = dataset[dataset["movimiento"].isin(CLASES)].reset_index(drop=True)
    return dataset


def seleccionar_caracteristicas(
    X_df: pd.DataFrame,
    y: np.ndarray,
    max_features: int = 12,
    corr_threshold: float = 0.90,
) -> list[str]:
    """Filtra características mediante ANOVA (p < 0.05) y elimina redundancia (|r| >= 0.90)."""
    resultados_anova = []
    clases_unicas = np.unique(y)

    for col in FEATURE_NAMES:
        grupos = [X_df.loc[y == c, col].to_numpy() for c in clases_unicas]
        f_stat, p_val = f_oneway(*grupos)
        if np.isnan(f_stat):
            f_stat, p_val = 0.0, 1.0
        resultados_anova.append((col, f_stat, p_val))

    df_anova = pd.DataFrame(resultados_anova, columns=["feature", "f_stat", "p_val"])
    significativas = df_anova[df_anova["p_val"] < 0.05].sort_values("f_stat", ascending=False)

    if len(significativas) < 4:
        significativas = df_anova.sort_values("f_stat", ascending=False)

    candidatas = significativas["feature"].tolist()
    matriz_corr = X_df[candidatas].corr().abs()

    seleccionadas = []
    for feat in candidatas:
        if all(matriz_corr.loc[feat, sel] < corr_threshold for sel in seleccionadas):
            seleccionadas.append(feat)
        if len(seleccionadas) >= max_features:
            break

    return seleccionadas


def obtener_modelos() -> dict:
    return {
        "SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, alpha=1e-3, random_state=42),
    }


def guardar_matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, nombre_modelo: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=CLASES)

    plt.figure(figsize=(6.5, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASES,
        yticklabels=CLASES,
    )
    plt.title(f"Matriz de Confusión LOSO - {nombre_modelo}")
    plt.xlabel("Actividad Predicha")
    plt.ylabel("Actividad Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"matriz_confusion_{nombre_modelo}.png", dpi=300)
    plt.close()


def main() -> None:
    print("1. Cargando señales y extrayendo características de ventanas...")
    dataset = construir_dataset_ventanas(DATA_RAW_DIR)
    sujetos = dataset["sujeto"].to_numpy()
    y = dataset["movimiento"].to_numpy()
    X_full = dataset[FEATURE_NAMES]

    n_sujetos = len(np.unique(sujetos))
    print(f"   -> Total de ventanas: {len(dataset)} | Sujetos detectados ({n_sujetos}): {sorted(np.unique(sujetos))}")
    print(f"   -> Distribución de clases:\n{dataset['movimiento'].value_counts().to_string()}\n")

    if n_sujetos >= 2:
        cv = LeaveOneGroupOut()
        splits = list(cv.split(X_full, y, groups=sujetos))
        esquema_eval = "LOSO"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X_full, y))
        esquema_eval = "StratifiedKFold_5"
        print("   [Aviso] Solo se detectó 1 sujeto; usando validación cruzada de 5 pliegues.\n")

    modelos = obtener_modelos()
    resultados_modelos = []
    predicciones_por_modelo = {}

    print(f"2. Evaluando modelos con validación {esquema_eval}...")
    for nombre, _ in modelos.items():
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in splits:
            X_train_df = X_full.iloc[train_idx]
            y_train = y[train_idx]
            X_test_df = X_full.iloc[test_idx]
            y_test = y[test_idx]

            cols_fold = seleccionar_caracteristicas(X_train_df, y_train, max_features=12)

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_df[cols_fold])
            X_test_sc = scaler.transform(X_test_df[cols_fold])

            clf = obtener_modelos()[nombre]
            clf.fit(X_train_sc, y_train)
            preds = clf.predict(X_test_sc)

            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)
        predicciones_por_modelo[nombre] = (y_true_arr, y_pred_arr)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        f1_m = f1_score(y_true_arr, y_pred_arr, average="macro")
        kappa = cohen_kappa_score(y_true_arr, y_pred_arr)

        resultados_modelos.append(
            {
                "modelo": nombre,
                "esquema_validacion": esquema_eval,
                "accuracy": round(acc, 4),
                "f1_macro": round(f1_m, 4),
                "cohen_kappa": round(kappa, 4),
            }
        )
        print(f"   - {nombre:<14} | Accuracy: {acc:.4f} | F1-macro: {f1_m:.4f} | Kappa: {kappa:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_resultados = pd.DataFrame(resultados_modelos).sort_values("f1_macro", ascending=False)
    df_resultados.to_csv(RESULTS_DIR / "metricas_loso.csv", index=False)

    mejor_modelo_nombre = df_resultados.iloc[0]["modelo"]
    y_true_best, y_pred_best = predicciones_por_modelo[mejor_modelo_nombre]
    guardar_matriz_confusion(y_true_best, y_pred_best, mejor_modelo_nombre)

    print(f"\n3. Entrenando modelo final ({mejor_modelo_nombre}) con todo el dataset...")
    cols_finales = seleccionar_caracteristicas(X_full, y, max_features=12)
    scaler_final = StandardScaler()
    X_final_sc = scaler_final.fit_transform(X_full[cols_finales])

    modelo_final = obtener_modelos()[mejor_modelo_nombre]
    modelo_final.fit(X_final_sc, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artefacto = {
        "model": modelo_final,
        "scaler": scaler_final,
        "selected_features": cols_finales,
        "classes": CLASES,
        "fs": FS,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "best_model_name": mejor_modelo_nombre,
    }
    joblib.dump(artefacto, MODEL_PATH)

    print(f"   -> Características seleccionadas ({len(cols_finales)}): {cols_finales}")
    print(f"   -> Modelo exportado correctamente en: {MODEL_PATH}")
    print(f"   -> Métricas guardadas en: {RESULTS_DIR / 'metricas_loso.csv'}")


if __name__ == "__main__":
    main()