# Análisis de Actividad Electrodérmica (EDA) - Dataset WESAD

Esta carpeta contiene el pipeline definitivo para el procesamiento, extracción de características y clasificación de estrés utilizando la señal de Actividad Electrodérmica (EDA) del dataset WESAD (Wearable Stress and Affect Detection).

Este trabajo forma parte de un proyecto más grande de **fusión multimodal** junto con señales de EMG y RESP.

## Estructura del Proyecto

El código está dividido en dos fases principales:

### 1. `EDA_v3.py` (Extracción y Limpieza)
Procesa los datos crudos, sincroniza las ventanas de tiempo del pecho (RespiBAN) y la muñeca (Empatica E4), aplica filtros de artefactos y extrae 22 características relevantes.

* **Entrada:** Archivos originales sincronizados del WESAD en formato `.pkl` (ej. `S2.pkl`, `S3.pkl`). De estos archivos, el script extrae y procesa estrictamente tres señales:
  * EDA del pecho (sensor RespiBAN a 700 Hz).
  * EDA de la muñeca (sensor Empatica E4 a 4 Hz).
  * Acelerómetro tridimensional (ACC) de la muñeca (Empatica E4 a 32 Hz).
* **Salida:** `EDA_v3_todos.csv` (105,993 ventanas síncronas de 15 sujetos).

### 2. `loso_final.py` (Evaluación Machine Learning)
Toma las características extraídas y entrena modelos (Random Forest, AdaBoost, LDA) usando validación cruzada estricta *Leave-One-Subject-Out* (LOSO).

* **Entrada:** `EDA_v3_todos.csv`
* **Salidas:** Métricas de rendimiento, matrices de confusión y predicciones probabilísticas fuera de muestra (`.csv.gz`).

## Ejecución de los Scripts (Opcional)

**Nota Importante:** Los datasets finales (`EDA_v3_todos.csv`) y todas las métricas de validación ya se encuentran generados en sus respectivas carpetas. **No es necesario volver a ejecutar el código** para utilizar los datos en la integración con otras señales. Los scripts `.py` se incluyen exclusivamente con fines de reproducibilidad científica y validación metodológica.

 **Aviso de rendimiento:** El procesamiento de las señales EDA y la validación cruzada LOSO conllevan una alta carga computacional. Volver a ejecutar el flujo completo tomará varios minutos, por lo que se recomienda fuertemente el uso de procesamiento en paralelo (ej. Kaggle o equipos multinúcleo).

Si deseas ejecutar el código desde cero para realizar pruebas adicionales, asegúrate de instalar las dependencias requeridas primero:

```bash
pip install -r requirements_EDA.txt
```

Luego, puedes ejecutar los scripts. Se recomienda usar el argumento `--procesos` para aprovechar múltiples núcleos:

**Fase 1: Generar el dataset final**
```bash
python EDA_v3.py --ruta /ruta/a/WESAD --salida ./ --procesos 4
```

**Fase 2: Ejecutar los experimentos ML y obtener métricas**
```bash
python loso_final.py --tabla EDA_v3_todos.csv --salida resultados_loso_final --procesos 4
```

##  Anotaciones para la Fusión Multimodal (Archivos CSV de EMG y RESP)

Para unir este trabajo con los modelos de las otras señales fisiológicas, utilicen los siguientes archivos ubicados en la carpeta de salidas:

* **Características directas:** El archivo `EDA_v3_todos.csv` contiene la matriz de características. Usen las columnas `sujeto` y `t_inicio` como claves primarias (*primary keys*) para alinear sus ventanas de tiempo. Las etiquetas y datos de calidad no deben incluirse en el entrenamiento.
* **Fusión Tardía (Stacking):** En la carpeta `resultados_loso_final/predicciones/` encontrarán archivos como `3clases_ambos.csv.gz`. Estos contienen las probabilidades predictivas por ventana (sin fugas de información gracias al esquema LOSO) listas para ser promediadas o fusionadas con sus predicciones.