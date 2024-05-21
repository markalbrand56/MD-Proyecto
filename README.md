# Explorando los Determinantes de la Depresión: Un Análisis Global

---

**Karen Jimena Hernández Ortega**  
**Mark Alexander Albrand Mendoza**  
**Mario Cristales**  
**Javier Heredia**

Departamento de Ciencias de la Computación. Facultad de Ingeniería, Universidad del Valle de Guatemala
**Luis Roberto Furlan Collver**  

---

## Análisis de Datos

### Características y Procesamiento de los Datos

El conjunto de datos *Mental Health Depression Disorder Data* incluye variables relacionadas con los desórdenes de depresión, organizadas en seis hojas de un documento de Excel. Las hojas contienen información sobre el uso de sustancias, nivel de educación, edad, género, conteo total por país de la prevalencia de la depresión y tasas de suicidio.

Después de una rigurosa exploración de datos la cual consistió en analizar cada una de las hojas del documento Excel, se realizó el siguiente proceso para cada una, el cual se presenta detallado en el [notebook de exploración](https://github.com/markalbrand56/MD-Proyecto/blob/main/proyecto-g6.ipynb):

- 1. Verifación de valores nulos
- 2. Generación de ProfileReport, en donde se analizó cada una de las features para esa hoja.
- 3. Limpieza de datos, en donde se optó por eliminar los datos faltantes, por diferentes razones explicadas en el [notebook de exploración](https://github.com/markalbrand56/MD-Proyecto/blob/main/proyecto-g6.ipynb)
- 4. Separación de variables cuantitativas y cualitativas, en donde para cada una se analizó el tipo de dato, se analizó la dispersión de los datos por medio de diagramas de caja y bigotes y se generó un histrograma para representar la distribución de cada una.
- 5. Se analizó la correlación para cada una de las variables. Para determinar que tan relacionadas se encontraban entre sí.

También, se seleccionó *“Depressive disorder rates (number suffering per 100,000)”* como la variable objetivo, 


Además, se eligieron 24 variables adicionales que se consideraron relevantes para el análisis.

A partir de esto, se realizó una matriz de correlación con la variables elegidas que puede ser encontrada en el notebook [correlation.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/correlation.ipynb). Por último, se exportó un archivo CSV con las variables seleccionadas.

Para el procesamiento de los datos, se realizaron procesos de limpieza, escalado y optimización de parámetros. Los datos hasta 2016 se utilizaron para entrenamiento (80%) y prueba (20%), y los datos de 2017 en adelante para calculo de predicciones.

### Algoritmos Utilizados

#### KNN

Se utilizó el algoritmo de agrupamiento K-means para definir los clusters de los datos. Se utilizó la biblioteca Scikit-learn para la implementación del algoritmo. En el notebook [clustering.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/clustering.ipynb) se puede observar el proceso de división de los datos, la creación del modelo y la predicción de los clusters. Luego de analizar la mejor cantidad de clusters, se encontró que el número óptimo era 10. Los resultados se muestran en la siguiente tabla:

| Cluster | Cantidad de datos |
|---------|--------------------|
| 1       | 1094               |
| 2       | 151                |
| 3       | 1299               |
| 4       | 28                 |
| 5       | 28                 |
| 6       | 214                |
| 7       | 623                |
| 8       | 354                |
| 9       | 504                |
| 10      | 1193               |

Los grupos más notables fueron el primero, el tercero y el décimo. El primer grupo tenía las menores tasas de esquizofrenia, desórdenes alimenticios y uso de drogas. El tercer grupo tenía las menores tasas en variables de edad y género. El décimo grupo tenía el promedio más bajo en desórdenes bipolares. Estos resultados son útiles para identificar patrones y características en los datos.

#### Red Neuronal

Las redes neuronales, inspiradas en el funcionamiento del sistema nervioso humano, son adecuadas para manejar conjuntos de datos complejos y de alta dimensión. La red neuronal utilizada en este estudio tenía siete capas, incluyendo capas convolucionales, de max pooling, aplanado y densas. Se utilizó la biblioteca Keras de TensorFlow para la implementación de la red. Esta red se encuentra en el archivo [red_neuronal_entreno.py](https://github.com/markalbrand56/MD-Proyecto/blob/main/red_neuronal_entreno.py).

```python
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),  # Convolutional layer: 64 filtros de 3x1 para extraer características
    MaxPooling1D(2),  # Max pooling layer: Reducir la dimensión de las características para obtener las más importantes
    Conv1D(64, 3, activation='relu'),  # Convolutional layer: 64 filtros de 3x1 para extraer características
    MaxPooling1D(2),  # Max pooling layer: Reducir la dimensión de las características para obtener las más importantes
    Flatten(),  # Aplanar las características para poder conectarlas a una capa densa
    Dense(64, activation='relu'),  # Capa densa con 64 neuronas y función de activación ReLU
    Dense(1)  # Capa densa con 1 neurona para la regresión
])
```

En el archivo se puede observar el proceso de división de los datos, la creación del modelo, la compilación y el entrenamiento del modelo. Se utilizó el optimizador Adam y las funciones de pérdida fueron el error cuadrático medio (MSE) y el error absoluto medio (MAE).

La red neuronal se entrenó con un total de 20,000 iteraciones en un proceso que duró aproximadamente dos horas en completarse. El equipo utilizado contaba con las siguientes características: Ryzen 9 7940HS de 8 núcleos/16 hilos y 32 GB de memoria RAM.

#### Máquina de Vectores de Soporte (SVM)

El SVM es un algoritmo supervisado que busca encontrar el hiperplano óptimo para clasificar datos en un espacio N-dimensional. Se utilizó un kernel lineal ya que los datos eran linealmente separables. Se creó una nueva variable *Depression_binary* para clasificar las tasas de depresión en tres categorías: bajo, medio y alto. Se utilizó la biblioteca Scikit-learn para la implementación del SVM. El procesamiento de los datos y la creación del modelo se encuentran en el notebook [SVM.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/SVM.ipynb). Se utilizó la siguiente implementación:

```python
from sklearn.svm import SVC
clasificador = SVC(kernel = 'linear', random_state = 0)
clasificador.fit(X_entreno, y_entreno)

y_pred = clasificador.predict(X_prueba)

```

En el notebook se puede observar el proceso de división de los datos, la creación del modelo, la predicción y la evaluación del modelo. Se utilizó la matriz de confusión y las métricas de precisión, recall, F1-score y accuracy para evaluar el rendimiento del modelo. El modelo se entrenó aproximadamente en menos de un segundo. El equipo usado contaba con las siguientes características: MacBook M2 Pro, 2023 de 16 GB de memoria RAM.


## Resultados y discusión

### Exploración de los Datos

Se realizó un análisis exploratorio de los datos para identificar patrones y características. Se encontró que hay un total de 196 países diferentes con datos desde 1990 hasta 2017, distribuidos uniformemente.

A partir del análisis realizado sobre el conjunto de datos por medio de cada ProfileReport, se revelaron hallazgos detallados sobre la prevalencia de la depresión. La tasa máxima de depresión registrada es del 6.6%, mientras que la media global es del 3.49%. Al desglosar los datos por grupos de edad, se observan diferencias significativas: la media de depresión en personas de 10 a 14 años es del 1.37%, en el grupo de 20 a 24 años es del 3.79%, y en aquellos mayores de 70 años, la media es significativamente mayor, alcanzando el 6.13%. Al analizar las diferencias de género, se confirma una mayor prevalencia de mujeres, con una media del 4.16% en comparación con el 2.80% en hombres.

#### Red Neuronal

La red neuronal mostró un error absoluto medio (MAE) y un error cuadrático medio (MSE) durante las fases de entrenamiento y evaluación, indicando un buen ajuste a los datos. Sin embargo, los datos de validación mostraron errores más elevados, lo cual era esperado debido a la falta de exposición del modelo a estos datos durante el entrenamiento.

| Fase          | MAE    | MSE          |
|---------------|--------|--------------|
| Entrenamiento | 2.3356 | 10.0856      |
| Evaluación    | 2.3274 | 9.9416       |
| Predicciones  | 468.5533 | 351,443.73489 |

El modelo de red neuronal mostró un rendimiento aceptable en la predicción. En la fase de predicciones, en el que se utilizaron datos de 2017 en adelante, el modelo mostró un error absoluto medio de 468.5533 y un error cuadrático medio de 351,443.73489. Esto muestra que en promedio el error de la predicción es aceptable, pero que existirán casos en los que la predicción será muy distante del valor real, lo cual es esperado en el modelo. Por lo cual se considera como satisfactorio el rendimiento del modelo.

En el notebook [red_neuronal.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/red_neuronal.ipynb) se encuentra la evaluación de las predicciones con los datos de 2017 en adelante.

#### SVM

El SVM mostró un rendimiento excepcional en la clasificación de los datos, con altas métricas de precisión, recall, F1-score y accuracy.

| Fase          | Precision | Recall | F1-score | Accuracy |
|---------------|-----------|--------|----------|----------|
| Evaluación    | 0.989     | 0.99   | 0.99     | 0.99     |
| Predicciones  | 0.994     | 0.99   | 0.99     | 0.99     |

Como se observa, los resultados tanto en la fase de entrenamiento como de validación son bastante altos, lo que indica que los modelos son altamente eficaces y confiables para la detección de desórdenes de depresión en poblaciones amplias.

## Conclusiones

- Los clusters identificados por el algoritmo K-means mostraron patrones y características claras en los datos.
- La red neuronal y el SVM mostraron buenos resultados en la predicción de la depresión,siendo capaces de clasificar los datos con alta precisión.deser eficaces

## Referencias

- Suriano, A. (2024). Naive Bayes & Laplace smoothing, SVM, Arboles de Decision. Departamento de Ciencias de la Computación y Tecnologías de la Información, Universidad del Valle de Guatemala.
- Carrillo García, S. (2019). Artículo científico. En S. Carrillo García, L. M. Toro Calderón, A. X. Cáceres González y E. C. Jiménez Lizarazo, Caja de herramientas. Géneros Textuales. Universidad Santo Tomás.
- CRAI USTA Bucaramanga. (2020). Informe de recursos y servicios bibliográficos. Universidad Santo Tomás.
- Vargas Leal, V. M., Galvis García, R. E., Idárraga Ortiz, S. A. y López Báez, J. D. (2020). Guía resumen del estilo APA: séptima edición. Universidad Santo Tomás. http://hdl.handle.net/11634/34384
- IBM. (2021, 17 agosto). El modelo de redes neuronales. https://www.ibm.com/docs/es/spss-modeler/saas?topic=networks-neural-model
- Del Cid, M. T. C. (2021). La depresión y su impacto en la salud pública. Revista médica hondureña, 89(Supl. 1), 46-52.
- Muñoz, V., Alvarado, C. L. A., Barros, J. M. T., & Malla, M. I. M. (2021). Prevalencia de depresión y factores asociados en adolescentes: Artículo original. Revista Ecuatoriana de Pediatría, 22(1), 6-1.
- Organización Mundial de la Salud (2023, Marzo 31). Depresión. Organización Mundial de la Salud. https://www.who.int/es/news-room/fact-sheets/detail/depression#:~:text=Se%20estima%20que%20el%203,personas%20sufren%20depresi%C3%B3n%20(1).
