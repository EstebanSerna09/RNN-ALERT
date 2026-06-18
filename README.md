# RNN-ALERT

Sistema de predicción de deserción estudiantil y estimación de rendimiento académico mediante modelos de Deep Learning y Machine Learning.

El proyecto permite ejecutar pipelines de limpieza, preparación, entrenamiento, predicción y comparación de modelos para apoyar el análisis académico de estudiantes.

## 1. Descripción general

RNN-ALERT integra dos enfoques principales de modelado:

1. **RNN Multitarea**

   * Modelo principal basado en una arquitectura multimodal.
   * Usa historia académica secuencial y variables sociodemográficas.
   * Realiza dos tareas:

     * Clasificación binaria de deserción.
     * Regresión de nota.

2. **Modelos Machine Learning**

   * Modelos clásicos entrenados por programa académico.
   * Se dividen en dos bloques:

     * Clasificación de deserción con datos sociodemográficos.
     * Regresión de nota con datos académicos.

## 2. Estructura general del proyecto

```text
rnn_alert/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modelos/
├── pipelines/
├── utils/
├── assets/
├── data/
├── outputs/
│   ├── modelos/
│   ├── modelos_ml/
│   ├── datasets/
│   └── resultados/
```

## 3. Carpetas principales

### `modelos/`

Contiene la lógica de entrenamiento y definición de modelos, incluyendo:

* RNN Multitarea.
* Modelos Machine Learning.
* Entrenamiento por programa.
* Métricas y evaluación.

### `pipelines/`

Contiene los procesos de limpieza, transformación y preparación de datos.

Los pipelines permiten generar datasets procesados para entrenamiento y predicción.

### `utils/`

Contiene funciones auxiliares del proyecto, como:

* Manejo de sesión.
* Persistencia de modelos.
* Carga de modelos guardados.
* Descarga y carga de datasets procesados.
* Funciones de apoyo para Streamlit.

### `data/`

Carpeta donde deben ubicarse los datos originales o archivos base necesarios para ejecutar los pipelines.

Si el proyecto se entrega sin datos por motivos de privacidad o peso, el usuario debe colocar los datasets originales en esta carpeta antes de ejecutar los pipelines.

### `outputs/`

Carpeta donde el sistema genera automáticamente los resultados.

Esta carpeta inicia vacía en la versión limpia del proyecto.

Subcarpetas:

```text
outputs/modelos/
```

Guarda modelos RNN entrenados.

```text
outputs/modelos_ml/
```

Guarda Modelos Machine Learning entrenados por programa.

```text
outputs/datasets/
```

Guarda datasets procesados por los pipelines.

```text
outputs/resultados/
```

Guarda métricas, reportes y resultados de entrenamiento.

## 4. Instalación en una máquina nueva

### 4.1. Entrar a la carpeta del proyecto

```powershell
cd ruta\donde\se\descomprimio\rnn_alert
```

Ejemplo:

```powershell
cd "C:\Users\Usuario\Desktop\rnn_alert"
```

### 4.2. Crear entorno virtual

```powershell
python -m venv venv
```

### 4.3. Activar entorno virtual

```powershell
venv\Scripts\activate
```

### 4.4. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 4.5. Ejecutar la aplicación

```powershell
streamlit run app.py
```

## 5. Flujo de uso recomendado

Después de abrir la aplicación en Streamlit, se recomienda seguir este orden:

1. Cargar o importar los datasets originales.
2. Ejecutar el pipeline de limpieza.
3. Ejecutar los pipelines de transformación y preparación.
4. Verificar o descargar los datasets procesados.
5. Entrenar la **RNN Multitarea**.
6. Entrenar los **Modelos Machine Learning**.
7. Revisar métricas.
8. Ejecutar predicción individual o masiva.
9. Revisar la comparación de modelos.

## 6. Modelos disponibles

### 6.1. RNN Multitarea

La RNN Multitarea trabaja con dos entradas:

1. Historia académica secuencial.
2. Variables sociodemográficas.

Y genera dos salidas:

1. Probabilidad de deserción.
2. Predicción de nota.

### 6.2. Modelos Machine Learning

Los Modelos Machine Learning se entrenan por programa académico.

#### Clasificación de deserción con datos sociodemográficos

Modelos:

* SVM Classifier.
* Random Forest Classifier.
* Regresión Logística.

Métricas principales:

* Accuracy.
* Precision.
* Recall.
* F1-score.
* AUC.
* PR-AUC, si aplica.

#### Regresión de nota con datos académicos

Modelos:

* SVM Regressor.
* Random Forest Regressor.
* Regresión Lineal.

Métricas principales:

* MAE.
* MSE.
* RMSE.
* R2.

## 7. Persistencia de archivos generados

La versión limpia del proyecto no incluye modelos entrenados ni datasets procesados.

Cuando el usuario ejecute los pipelines y entrenamientos, el sistema generará automáticamente archivos en:

```text
outputs/modelos/
outputs/modelos_ml/
outputs/datasets/
outputs/resultados/
```

Estos archivos no se incluyen en la versión limpia para reducir el peso del proyecto.

## 8. Archivos que no se deben subir ni enviar

No se deben incluir en GitHub ni en el ZIP de entrega:

```text
venv/
.venv/
env/
__pycache__/
outputs/modelos/
outputs/modelos_ml/
outputs/datasets/
outputs/resultados/
logs/
runs/
checkpoints/
*.keras
*.h5
*.pkl generados
*.joblib
*.sav
*.ckpt
*.log
```

## 9. Cómo dejar el proyecto limpio nuevamente

Desde la carpeta padre del proyecto se puede ejecutar el script:

```powershell
.\limpiar_rnn_alert.ps1
```

Este script elimina:

* Entornos virtuales.
* Modelos entrenados.
* Datasets procesados.
* Métricas generadas.
* Cachés.
* Logs.
* Archivos temporales.

Y genera un ZIP limpio del proyecto.

## 10. Nota importante

Si se entrega el proyecto sin datasets originales, la persona que lo reciba deberá colocar los archivos base en la carpeta correspondiente antes de ejecutar los pipelines.

Si se entrega con datasets originales, deben ubicarse en:

```text
data/
```

o en la ruta que indique la interfaz de carga de datos del proyecto.

## 11. Ejecución rápida

```powershell
cd "ruta\del\proyecto\rnn_alert_v4"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 12. Resultado esperado

Al ejecutar el proyecto limpio:

1. La aplicación abre correctamente en Streamlit.
2. No hay modelos entrenados cargados inicialmente.
3. El usuario debe ejecutar los pipelines.
4. El usuario debe entrenar por primera vez la RNN Multitarea.
5. El usuario debe entrenar los Modelos Machine Learning.
6. Los resultados se generan nuevamente dentro de `outputs/`.
