# Reconocimiento de Actividad Humana (HAR) con Sensor Inercial WT9011DCL

Sistema de adquisición, procesamiento de señales inerciales y clasificación automática de actividades locomotoras humanas (`caminar`, `brincar` y `subir_escaleras`) desarrollado para el **Proyecto Integrativo II** (Ingeniería Biomédica).

---

## Estructura del Repositorio

```text
HAR-WT9011DCL-Biomedica/
├── data/
│   ├── raw/                  # Grabaciones experimentales por sujeto y repetición (.csv)
│   └── test_samples/         # Muestras de validación rápida (.csv)
├── models/
│   └── har_pipeline.joblib   # Modelo entrenado, StandardScaler y características seleccionadas
├── results/
│   ├── figuras_articulo/     # Matriz de confusión LOSO y gráficas de resultados
│   └── metricas_loso.csv     # Tabla comparativa de desempeño entre clasificadores
├── src/
│   ├── config.py             # Constantes de adquisición (100 Hz), ventanas y rutas
│   ├── sensor_ble.py         # Driver Bluetooth Low Energy (BLE) para WitMotion WT9011DCL
│   ├── processing.py         # Carga de archivos, segmentación y extracción de 48 características
│   └── train_evaluate.py     # Selección ANOVA + correlación, validación LOSO y exportación
├── requirements.txt          # Dependencias del proyecto
└── README.md