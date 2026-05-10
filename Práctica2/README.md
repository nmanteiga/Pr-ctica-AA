# Práctica 2: Lenguaje de Signos Español (LSE)
Este repositorio contiene la resolución de la Práctica de Aprendizaje Automático enfocada en el reconocimiento de letras del Lenguaje de Signos Español mediante características geométricas de las manos (Landmarks) y visión por computador.

## Estructura del Proyecto
* **dataset/**: Carpeta que contiene las imágenes organizadas por letras, así como el archivo procesado `landmarks_lse.csv` del cual leen los modelos.
* **fonts/**: Código de la práctica anterior o utilidades dadas por el equipo docente, como la validación cruzada y extracción de métricas (`soluciones.jl` y `soluciones2.j* **fonts/**: Código d*:* **fonts/**: Código de la práctica anterior o utilidades dadas por el equipo docente, como la  ** **fonts/**: Código de la práctica anterior o utilidades dadas por el equipo docente, como la validación cruzada y extracción de métricas (`soluciones.jl` y `soluciones2.j* **fonts/**: Código d*:* **fonts/**: Cript de prueba (opcional).
* **fonts/**: en esta carpeta hay un script en python que nos ayuda a procesar el dataset (generar_dataset.py, no hace falta ejecutarse para la evaluación ya que ya ha sido ejecutado previamente) y dos scripts que hacen que el usuario pueda visualizar a través de la cámara de su ordenador cómo funciona el modelo (entrena el modelo con el dataset y luego analiza las letras mediante las posiciones geométricas en directo, para esto usar: lse_realtime.jl). 

## Aproximaciones Evaluadas (Scripts Principales)
Los siguientes archivos deben ser ejecutables desde Julia vía `include("nombre_archivo.jl")`:

* **arbolesDecision.jl**: Implementación y evaluación usando Árboles de Decisión (DecisionTreeClassifier).
* **knn.jl**: Implementación usando K-Nearest Neighbors (KNeighborsClassifier).
* **svm.jl**: Implementación con Máquinas de Vectores de Soporte (SVC / linear).
* **rrnnaa.jl**: Aproximación mediante Redes Neuronales Artificiales (ANN).
* **deeplearning.jl**: Modelo con aprendizaje profundo para procesamiento o capas convolucionales.
* **dome.jl**: Aproximación clásica utilizando SymDoME.
