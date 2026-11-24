from pathlib import Path

# Rutas principales del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_TEST_DIR = BASE_DIR / "data" / "test_samples"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figuras_articulo"
MODEL_PATH = MODELS_DIR / "har_pipeline.joblib"

# Configuración BLE del sensor WitMotion WT9011DCL
DEVICE_NAME_PREFIX = "WT9011DCL"
UUID_DATOS = "0000ffe4-0000-1000-8000-00805f9a34fb"
PACKET_HEADER = b"\x55\x61"
PACKET_LEN = 20

# Parámetros de adquisición y segmentación
FS = 50                        # Frecuencia de muestreo (Hz)
WINDOW_SECONDS = 2              # Duración de ventana (s)
WINDOW_SIZE = FS * WINDOW_SECONDS  # 200 muestras
OVERLAP = 0.5                   # 50% de traslape
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 100 muestras (1 s)
DISCARD_INITIAL_SAMPLES = 100   # Descarte del primer segundo en tomas largas

# Columnas esperadas y clases de movimiento
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
CLASES = ["caminar", "brincar", "subir_escaleras"]
