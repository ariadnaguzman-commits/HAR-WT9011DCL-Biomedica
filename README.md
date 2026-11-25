# Reconocimiento de actividad humana (HAR) con sensor inercial WT9011DCL

Sistema de adquisición, procesamiento de señales inerciales y clasificación automática de actividades locomotoras humanas (caminar, brincar y subir escaleras), desarrollado para el Proyecto Integrativo II de Ingeniería Biomédica.

## 1. Estructura del repositorio

```text
HAR-WT9011DCL-Biomedica/
│
├── data/
│   ├── raw/                         # Grabaciones experimentales por sujeto y repetición (.csv).
│   └── test_samples/                # Muestras para validación rápida (.csv).
│
├── models/
│   └── har_pipeline.joblib          # Modelo entrenado, escalador y características seleccionadas.
│
├── results/
│   ├── figuras_articulo/            # Matriz de confusión LOSO y diagramas de caja.
│   ├── metricas_loso.csv            # Desempeño global por modelo y conjunto de características.
│   ├── metricas_por_clase.csv       # Precisión, recall y F1 por actividad.
│   └── ranking_seleccion_caracteristicas.csv
│                                    # Ranking ANOVA F-score e importancia de Random Forest.
│
├── src/
│   ├── config.py                    # Configuración centralizada del proyecto.
│   ├── acquisition.py               # Protocolo experimental de adquisición de datos.
│   ├── sensor_ble.py                # Comunicación y lectura del sensor mediante BLE.
│   ├── processing.py                # Preprocesamiento y extracción de características.
│   └── train_evaluate.py             # Entrenamiento, evaluación e inferencia de modelos.
│
└── requirements.txt                 # Dependencias de Python.
```

### Descripción de los archivos principales

- **`src/config.py`:** centraliza las rutas del proyecto, la frecuencia de muestreo (50 Hz), el tamaño y traslape de las ventanas (2 s y 50 %), el descarte inicial (1 s), las clases y las constantes BLE del sensor WT9011DCL.

- **`src/acquisition.py`:** implementa el protocolo experimental de 27 tomas por sujeto (tres repeticiones para cada una de las nueve variantes de las actividades caminar, brincar y subir escaleras) y guarda los archivos `.csv` en `data/raw/`.

- **`src/sensor_ble.py`:** decodifica los paquetes de 20 bytes (`0x55 0x61`) del sensor e implementa los trabajadores BLE (`BLEScannerWorker` y `BLEStreamWorker`) para establecer la conexión y realizar lecturas en tiempo real.

- **`src/processing.py`:** lee archivos `.csv` o `.mat`, aplica ventaneo deslizante y calcula 48 características estadísticas en el dominio del tiempo (media, desviación estándar, mínimo, máximo, rango y RMS sobre los seis ejes y las magnitudes de aceleración y velocidad angular).

- **`src/train_evaluate.py`:** construye el conjunto de datos a partir de las ventanas de `data/raw/`, compara métodos de selección de características (ANOVA F-score e importancia de Random Forest), reduce el conjunto a ocho características mediante significancia estadística y filtrado de correlación, y evalúa los modelos mediante validación intersujeto Leave-One-Subject-Out (LOSO). También genera las métricas y figuras correspondientes, entrena el modelo final y permite realizar inferencias mediante el argumento `--predict`.

## 2. Metodología y pipeline de procesamiento

### 2.1. Adquisición de señales inerciales

Se utiliza un sensor inercial WT9011DCL configurado a una frecuencia de muestreo de 50 Hz, ajustada al ancho de banda de 46 Hz del sensor. La comunicación se realiza mediante Bluetooth Low Energy (BLE), utilizando el siguiente UUID:

```text
0000ffe4-0000-1000-8000-00805f9a34fb
```

Se registran las siguientes señales:

- **Aceleración triaxial:** \(a_x, a_y, a_z\), expresada en unidades de gravedad (\(g\)).
- **Velocidad angular triaxial:** \(g_x, g_y, g_z\), expresada en grados por segundo (\(°/s\)).

### 2.2. Preprocesamiento y segmentación

Se descarta el primer segundo de cada grabación (50 muestras a 50 Hz) para evitar efectos asociados al arranque y estabilización del sensor.

Posteriormente, se aplica un esquema de ventanas deslizantes con los siguientes parámetros:

| Parámetro | Valor |
|---|---:|
| Frecuencia de muestreo | 50 Hz |
| Duración de cada ventana | 2 s |
| Muestras por ventana | 100 |
| Traslape | 50 % |
| Paso entre ventanas | 1 s (50 muestras) |
| Descarte inicial | 1 s (50 muestras) |

### 2.3. Extracción y selección de características

Se calculan 48 características estadísticas en el dominio del tiempo a partir de las señales inerciales.

Para reducir la dimensionalidad, se comparan dos métodos de selección:

- **ANOVA F-score:** identifica características con diferencias estadísticamente significativas entre las clases (\(p < 0.05\)).
- **Importancia de Random Forest:** permite evaluar la contribución de cada característica a la clasificación.

Se seleccionan ocho características, considerando la significancia estadística y la eliminación de redundancia mediante correlación (\(|r| \geq 0.90\)).

### 2.4. Entrenamiento y validación

Se comparan tres algoritmos de clasificación:

- Support Vector Machine con kernel de función de base radial (SVM-RBF).
- Random Forest.
- Multilayer Perceptron (MLP).

La evaluación se realiza mediante validación cruzada intersujeto Leave-One-Subject-Out (LOSO), en la que se utiliza un sujeto distinto para prueba en cada iteración, mientras que los sujetos restantes se emplean para entrenamiento.

Se comparan dos configuraciones:

1. Clasificación utilizando las 48 características originales.
2. Clasificación utilizando las ocho características seleccionadas.

El desempeño se reporta mediante métricas globales y por clase (precision, recall y F1-score), considerando las tres actividades: caminar, brincar y subir escaleras. Esto permite analizar tanto el rendimiento general como las posibles confusiones entre actividades.

## 3. Guía de instalación y ejecución

Esta guía describe el procedimiento para configurar y ejecutar el proyecto desde cero, incluso si no se cuenta previamente con Python, Git o las dependencias necesarias.

### Paso 1. Instalar Python y Git

1. Ingresa a [python.org](https://www.python.org/downloads/) y descarga Python 3.10 o superior (por ejemplo, Python 3.12).

   **Importante para Windows:** durante la instalación, marca la casilla `Add python.exe to PATH` antes de seleccionar `Install Now`.

2. Ingresa a [git-scm.com](https://git-scm.com/downloads), descarga Git e instálalo utilizando las opciones predeterminadas.

3. Cierra las terminales o ventanas de VS Code que tengas abiertas y abre una nueva terminal (PowerShell o Windows Terminal en Windows).

4. Verifica que ambas herramientas se hayan instalado correctamente:

   ```bash
   python --version
   ```

   ```bash
   git --version
   ```

   Si aparecen los números de versión correspondientes (por ejemplo, Python 3.12.x y Git 2.x.x), la instalación fue exitosa.

   En macOS o Linux, si el comando `python` no está disponible, utiliza `python3 --version`.

### Paso 2. Descargar el repositorio y configurar el entorno virtual

Un entorno virtual permite mantener aisladas las dependencias del proyecto, evitando conflictos con otros programas o proyectos de Python.

1. Abre una terminal y dirígete al directorio donde deseas guardar el proyecto. Por ejemplo, para utilizar el Escritorio en Windows:

   ```powershell
   cd Desktop
   ```

2. Clona el repositorio desde GitHub:

   ```bash
   git clone https://github.com/ariadnaguzman-commits/HAR-WT9011DCL-Biomedica.git
   ```

3. Accede al directorio del proyecto:

   ```bash
   cd HAR-WT9011DCL-Biomedica
   ```

4. Crea el entorno virtual:

   ```bash
   python -m venv .venv
   ```

5. Activa el entorno virtual.

   **En Windows (PowerShell):**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Si PowerShell muestra un error relacionado con los permisos o la política de ejecución, ejecuta el siguiente comando una sola vez:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

   Después, vuelve a activar el entorno virtual:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **En macOS o Linux:**

   ```bash
   python3 -m venv .venv
   ```

   ```bash
   source .venv/bin/activate
   ```

   Sabrás que el entorno está activo cuando aparezca `(.venv)` al inicio de la línea de comandos.

6. Actualiza `pip` e instala las dependencias del proyecto:

   ```bash
   python -m pip install --upgrade pip
   ```

   ```bash
   pip install -r requirements.txt
   ```

   Espera a que finalice la instalación de las bibliotecas necesarias, entre ellas NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn, Joblib y Bleak.

### Paso 3. Encender y configurar el sensor WT9011DCL

Para realizar nuevas adquisiciones, es necesario contar con el sensor inercial WT9011DCL.

1. Enciende el sensor y verifica que tenga batería suficiente y que su LED indique actividad.

2. Activa el Bluetooth de tu computadora.

   **Nota:** en Windows, mantén activado el Bluetooth del sistema, pero permite que el script establezca la conexión mediante BLE utilizando Bleak. Evita conectar previamente el sensor desde otra aplicación que pueda bloquear la comunicación.

3. Revisa los parámetros de conexión y muestreo en `src/config.py` y `src/acquisition.py`.

   Los parámetros principales son:

   - `DEFAULT_SENSOR_NAME`: nombre del sensor (por ejemplo, `WTSensor1`).
   - `DEFAULT_SENSOR_ADDRESS`: dirección MAC o identificador Bluetooth del sensor (por ejemplo, `FA:81:12:D8:44:D0`).
   - `UUID_DATOS`: UUID de la característica BLE utilizada para recibir los datos.
   - `FS`: frecuencia de muestreo configurada en 50 Hz.

4. Si utilizas un sensor físico diferente al configurado, modifica `DEFAULT_SENSOR_ADDRESS` o `DEFAULT_SENSOR_NAME` en `src/config.py` antes de realizar las adquisiciones.

### Paso 4. Realizar nuevas adquisiciones (opcional)

Este paso es necesario únicamente si deseas registrar un nuevo sujeto o repetir las mediciones del protocolo experimental. Si solamente deseas reproducir el entrenamiento con los datos disponibles en el repositorio, puedes omitirlo.

El protocolo contempla 27 tomas por sujeto: tres repeticiones para cada una de las nueve variantes de las actividades caminar, brincar y subir escaleras.

1. Abre `src/acquisition.py` y revisa la variable `SUJETO`, donde se especifica el identificador del participante (por ejemplo, `S07`).

2. Verifica las duraciones configuradas para cada actividad:

   | Actividad | Duración |
   |---|---:|
   | Caminar | 45 s |
   | Brincar | 30 s |
   | Subir escaleras | 30 s |

   Se contempla un segundo inicial adicional, que se descarta automáticamente durante el procesamiento.

3. Con el entorno virtual activo (`.venv`), el sensor encendido y ubicado cerca de la computadora, ejecuta:

   ```bash
   python src/acquisition.py
   ```

4. El programa mostrará el resumen de las 27 tomas, buscará el sensor mediante Bluetooth e indicará cuáles adquisiciones están pendientes.

5. Selecciona la toma que deseas realizar:

   - Introduce un número del 1 al 27 para realizar una toma específica.
   - Introduce `todas` para grabar las tomas pendientes en orden.
   - Introduce `0` para salir del programa.

6. Coloca el sensor en la cintura, de acuerdo con el protocolo experimental, y presiona Enter cuando estés listo.

7. Espera la cuenta regresiva (`3... 2... 1... ¡YA!`), realiza la actividad indicada y sigue las instrucciones de inicio y finalización de cada toma.

8. Al finalizar, el programa guardará automáticamente el archivo correspondiente en `data/raw/`, utilizando el siguiente formato de nombre:

   ```text
   SXX_NN_movimiento_variante_rR.csv
   ```

### Paso 5. Entrenar y evaluar los modelos

No es necesario tener el sensor encendido ni realizar nuevas adquisiciones para ejecutar este paso. El entrenamiento y la evaluación pueden reproducirse directamente con los archivos `.csv` disponibles en `data/raw/`.

Con el entorno virtual activo y situado en la raíz del repositorio `HAR-WT9011DCL-Biomedica`, ejecuta:

```bash
python src/train_evaluate.py
```

#### ¿Qué realiza este comando?

El script ejecuta las siguientes operaciones:

1. **Carga y preprocesamiento:** lee todos los archivos `.csv` disponibles en `data/raw/`, descarta el primer segundo de cada grabación y aplica ventanas deslizantes de 2 s (100 muestras), con un traslape del 50 % (paso de 50 muestras).

2. **Extracción de características:** calcula las 48 características estadísticas en el dominio del tiempo para cada ventana.

3. **Selección de características:** calcula los rankings mediante ANOVA F-score e importancia de Random Forest. Los resultados se guardan en:

   ```text
   results/ranking_seleccion_caracteristicas.csv
   ```

4. **Entrenamiento y evaluación:** compara los modelos SVM-RBF, Random Forest y MLP mediante validación intersujeto LOSO, evaluando tanto las 48 características originales como el conjunto reducido de ocho características.

5. **Cálculo de métricas:** imprime en consola las métricas globales y por clase (precision, recall y F1-score) para las actividades caminar, brincar y subir escaleras.

   Los resultados se guardan en:

   ```text
   results/metricas_loso.csv
   ```

   ```text
   results/metricas_por_clase.csv
   ```

6. **Entrenamiento del modelo final:** entrena el pipeline con las ocho características seleccionadas y guarda el modelo completo en:

   ```text
   models/har_pipeline.joblib
   ```

7. **Generación de figuras:** genera las visualizaciones correspondientes y las guarda en `results/figuras_articulo/`, incluyendo:

   - `matriz_confusion_Random_Forest.png`
   - `boxplots_top_caracteristicas.png`

### Paso 6. Realizar una predicción con un archivo de prueba

Una vez generado el modelo `models/har_pipeline.joblib`, es posible realizar una clasificación sobre un archivo de muestra sin necesidad de volver a entrenar todos los modelos.

Con el entorno virtual activo, ejecuta:

```bash
python src/train_evaluate.py --predict data/test_samples/S01_03_caminar_lento_r1.csv
```

El programa mostrará en la terminal los resultados de clasificación para cada ventana, incluyendo:

- `muestra_inicio`: índice de la muestra inicial de la ventana.
- `muestra_fin`: índice de la muestra final de la ventana.
- `prediccion`: actividad locomotora identificada por el modelo.

Si el archivo contiene una etiqueta real, también se reportará la exactitud obtenida en la prueba.

## 4. Tecnologías y bibliotecas utilizadas

El proyecto se desarrolla en Python y utiliza las siguientes herramientas:

| Tecnología | Aplicación |
|---|---|
| Python | Lenguaje de programación principal. |
| NumPy | Operaciones numéricas y procesamiento de señales. |
| Pandas | Lectura y manipulación de datos experimentales. |
| SciPy | Herramientas de procesamiento y análisis científico. |
| Scikit-learn | Selección de características, entrenamiento y evaluación de modelos. |
| Matplotlib | Generación de gráficas y visualizaciones. |
| Seaborn | Visualización estadística. |
| Joblib | Serialización y almacenamiento del modelo entrenado. |
| Bleak | Comunicación Bluetooth Low Energy con el sensor WT9011DCL. |
| Git y GitHub | Control de versiones y alojamiento del repositorio. |

## 5. Resultados

Los resultados del entrenamiento y la evaluación se encuentran organizados en el directorio `results/`.

Estos incluyen métricas de desempeño global, métricas desglosadas por actividad, rankings de selección de características y visualizaciones para analizar el comportamiento de los clasificadores.

La evaluación LOSO permite estudiar la capacidad de generalización de los modelos ante sujetos no observados durante el entrenamiento, proporcionando una estimación de su desempeño en participantes diferentes a los utilizados para ajustar cada modelo.