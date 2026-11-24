import sys
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
    precision_score,
    recall_score,
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


def comparar_metodos_seleccion(X_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Genera el ranking comparativo entre ANOVA F-score e Importancia de Random Forest."""
    clases_unicas = np.unique(y)
    f_scores, p_vals = [], []

    for col in FEATURE_NAMES:
        grupos = [X_df.loc[y == c, col].to_numpy() for c in clases_unicas]
        f_stat, p_val = f_oneway(*grupos)
        if np.isnan(f_stat):
            f_stat, p_val = 0.0, 1.0
        f_scores.append(f_stat)
        p_vals.append(p_val)

    rf_selector = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_selector.fit(X_df[FEATURE_NAMES], y)

    df_rank = pd.DataFrame(
        {
            "caracteristica": FEATURE_NAMES,
            "anova_f_score": np.round(f_scores, 4),
            "anova_p_val": np.round(p_vals, 6),
            "rf_importance": np.round(rf_selector.feature_importances_, 5),
        }
    )
    df_rank["rank_anova"] = df_rank["anova_f_score"].rank(ascending=False).astype(int)
    df_rank["rank_rf"] = df_rank["rf_importance"].rank(ascending=False).astype(int)
    return df_rank.sort_values("rank_anova").reset_index(drop=True)


def seleccionar_caracteristicas(
    X_df: pd.DataFrame,
    y: np.ndarray,
    max_features: int = 8,
    corr_threshold: float = 0.90,
) -> list[str]:
    """Filtra características mediante ANOVA (p < 0.05) y elimina redundancia (|r| >= 0.90)."""
    df_rank = comparar_metodos_seleccion(X_df, y)
    significativas = df_rank[df_rank["anova_p_val"] < 0.05]
    if len(significativas) < 4:
        significativas = df_rank

    candidatas = significativas["caracteristica"].tolist()
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


def guardar_figuras(
    dataset: pd.DataFrame,
    top_features: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Matriz de confusión del mejor modelo
    cm = confusion_matrix(y_true, y_pred, labels=CLASES)
    plt.figure(figsize=(6.5, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASES, yticklabels=CLASES)
    plt.title(f"Matriz de Confusión LOSO - {nombre_modelo}")
    plt.xlabel("Actividad Predicha")
    plt.ylabel("Actividad Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"matriz_confusion_{nombre_modelo}.png", dpi=300)
    plt.close()

    # 2. Boxplots de los 3 atributos más discriminantes (Sección II)
    feats_plot = top_features[:3]
    fig, axes = plt.subplots(1, len(feats_plot), figsize=(12, 4.2))
    if len(feats_plot) == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats_plot):
        sns.boxplot(data=dataset, x="movimiento", y=feat, order=CLASES, ax=ax, palette="Set2")
        ax.set_title(f"Distribución: {feat}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_top_caracteristicas.png", dpi=300)
    plt.close()


def predecir_archivo_prueba(ruta_archivo: str) -> None:
    """Ejemplo de inferencia reproducible usando el modelo serializado (Anexo E)."""
    artefacto = joblib.load(MODEL_PATH)
    df_raw = cargar_archivo(ruta_archivo)
    df_vent = procesar_dataframe(df_raw, window_size=artefacto["window_size"], step_size=artefacto["step_size"])
    X_sc = artefacto["scaler"].transform(df_vent[artefacto["selected_features"]])
    preds = artefacto["model"].predict(X_sc)
    df_vent["prediccion"] = preds

    print(f"\nArchivo evaluado: {ruta_archivo} ({len(df_vent)} ventanas)")
    print(df_vent[["muestra_inicio", "muestra_fin", "prediccion"]].to_string(index=False))
    if "movimiento" in df_vent.columns:
        acc = accuracy_score(df_vent["movimiento"], df_vent["prediccion"])
        print(f"Exactitud en archivo de prueba: {acc:.2%}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--predict":
        predecir_archivo_prueba(sys.argv[2])
        return

    print("1. Cargando señales y extrayendo características de ventanas...")
    dataset = construir_dataset_ventanas(DATA_RAW_DIR)
    sujetos = dataset["sujeto"].to_numpy()
    y = dataset["movimiento"].to_numpy()
    X_full = dataset[FEATURE_NAMES]

    n_sujetos = len(np.unique(sujetos))
    print(f"   -> Total de ventanas: {len(dataset)} | Sujetos ({n_sujetos}): {sorted(np.unique(sujetos))}")

    if n_sujetos >= 2:
        splits = list(LeaveOneGroupOut().split(X_full, y, groups=sujetos))
        esquema_eval = "LOSO"
    else:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_full, y))
        esquema_eval = "StratifiedKFold_5"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ranking = comparar_metodos_seleccion(X_full, y)
    df_ranking.to_csv(RESULTS_DIR / "ranking_seleccion_caracteristicas.csv", index=False)

    modelos = obtener_modelos()
    resultados_modelos = []
    predicciones_por_modelo = {}

    print(f"\n2. Evaluando modelos ({esquema_eval}): 48 features vs. 8 features podadas...")
    for conjunto, usar_poda in [("48_todas", False), ("8_seleccionadas", True)]:
        for nombre in modelos:
            y_true_all, y_pred_all = [], []

            for train_idx, test_idx in splits:
                X_train_df, y_train = X_full.iloc[train_idx], y[train_idx]
                X_test_df, y_test = X_full.iloc[test_idx], y[test_idx]

                cols = seleccionar_caracteristicas(X_train_df, y_train, max_features=8) if usar_poda else FEATURE_NAMES

                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train_df[cols])
                X_test_sc = scaler.transform(X_test_df[cols])

                clf = obtener_modelos()[nombre]
                clf.fit(X_train_sc, y_train)
                y_true_all.extend(y_test)
                y_pred_all.extend(clf.predict(X_test_sc))

            y_true_arr, y_pred_arr = np.array(y_true_all), np.array(y_pred_all)
            if usar_poda:
                predicciones_por_modelo[nombre] = (y_true_arr, y_pred_arr)

            acc = accuracy_score(y_true_arr, y_pred_arr)
            prec = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            rec = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            f1_m = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_true_arr, y_pred_arr)

            resultados_modelos.append(
                {
                    "modelo": nombre,
                    "conjunto_features": conjunto,
                    "esquema_validacion": esquema_eval,
                    "accuracy": round(acc, 4),
                    "precision_macro": round(prec, 4),
                    "recall_macro": round(rec, 4),
                    "f1_macro": round(f1_m, 4),
                    "cohen_kappa": round(kappa, 4),
                }
            )
            print(
                f"   - {nombre:<13} [{conjunto:<15}] | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1_m:.4f}"
            )

    df_resultados = pd.DataFrame(resultados_modelos)
    df_resultados.to_csv(RESULTS_DIR / "metricas_loso.csv", index=False)

    df_podadas = df_resultados[df_resultados["conjunto_features"] == "8_seleccionadas"].sort_values(
        "f1_macro", ascending=False
    )
    mejor_modelo_nombre = df_podadas.iloc[0]["modelo"]
    y_true_best, y_pred_best = predicciones_por_modelo[mejor_modelo_nombre]

    print(f"\n3. Entrenando modelo final ({mejor_modelo_nombre}) con 8 características...")
    cols_finales = seleccionar_caracteristicas(X_full, y, max_features=8)
    guardar_figuras(dataset, cols_finales, y_true_best, y_pred_best, mejor_modelo_nombre)

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
    print(f"   -> Características finales (8): {cols_finales}")
    print(f"   -> Artefacto guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()