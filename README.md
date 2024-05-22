# Explorando los Determinantes de la Depresión: Un Análisis Global

---

**Karen Jimena Hernández Ortega**  
**Mark Alexander Albrand Mendoza**  
**Mario Cristales**  
**Javier Heredia**

Departamento de Ciencias de la Computación. Facultad de Ingeniería, Universidad del Valle de Guatemala
**Luis Roberto Furlan Collver**  

---

## Resumen

La depresión es una condición compleja con causas diversas, desde factores sociales y psicológicos hasta influencias genéticas y biológicas. Este estudio utiliza el conjunto de datos Mental Health Depression Disorder Data para analizar la tasa de depresión por cada 100,000 habitantes en 196 países. La metodología incluyó la recolección y procesamiento de datos de un dataset que aborda variables como el uso de sustancias, nivel educativo, edad, género, prevalencia de la depresión por país y tasas de suicidio. Se seleccionaron 24 variables relevantes para el análisis, teniendo como variable objetivo Depressive disorder rates (number suffering per 100,000). Se utilizan dos algoritmos de aprendizaje automatizado: redes neuronales y máquinas de vectores de soporte. El primero se utiliza para realizar predicciones numéricas acerca de la variable objetivo, y el segundo para categorizar diferentes niveles de alerta. Los datos fueron divididos en conjuntos de entrenamiento y prueba siguiendo un enfoque temporal.  El modelo de red neuronal demostró sobresalir durante las fases de entrenamiento y prueba, aunque su precisión decayó al predecir con datos más recientes, pero manteniendo un rendimiento aceptable. El algoritmo de máquinas de vectores de soporte demostró una alta precisión en la clasificación de las tasas de depresión. Los resultados demuestran la efectividad de ambos modelos para predecir y clasificar la prevalencia de trastornos depresivos.

**Palabras Clave**: *depresión, análisis de datos, aprendizaje automático, redes neuronales, SVM*

## Introducción

La depresión es una condición compleja que puede ser causada por una amplia variedad de factores, que incluyen desde aspectos sociales y psicológicos hasta influencias genéticas y biológicas. Este trastorno mental, tan frecuente como desafiante, se caracteriza por una serie de síntomas que incluyen la pérdida de interés o placer en actividades cotidianas, sentimientos encontrados, falta de autoestima y una persistente sensación de tristeza (Muñoz et al., 2021). A medida que avanzan los años, se observa un aumento en el número de personas diagnosticadas con depresión en todo el mundo, afectando a diversos grupos etarios, desde adolescentes hasta adultos jóvenes y adultos mayores.

Los datos de la Organización Mundial de la Salud (OMS) brindan una perspectiva alarmante sobre la magnitud del problema: se estima que aproximadamente el 3,8% de la población mundial experimenta depresión, con una prevalencia aún mayor entre ciertos grupos demográficos. Por ejemplo, el 5% de los adultos, con una diferencia de género significativa, siendo del 4% en hombres y del 6% en mujeres, y el 5,7% de los adultos mayores de 60 años se ven afectados por este trastorno. Estas estadísticas globales revelan que alrededor de 280 millones de personas sufren de depresión en todo el mundo, subrayando la urgencia de abordar este problema de salud pública con seriedad y eficacia.

En este contexto, el conjunto de datos Mental Health Depression Disorder Data, es utilizado como una herramienta invaluable para la investigación y comprensión de los factores asociados a la depresión en diversos países. Este conjunto de datos abarca una amplia gama de variables relacionadas con los trastornos mentales, desde la adicción al consumo de sustancias hasta la prevalencia de la depresión por país y su correlación con variables como la educación, la edad y el género. 

El propósito de la realización de este artículo científico fue realizar un análisis de estos datos, poniendo especial énfasis en la variable "Depressive disorder rates (number suffering per 100,000)". Esta variable, que refleja la tasa de depresión por cada 100,000 habitantes por país, se convierte en el punto focal de la investigación. 

Se tienen dos objetivos específicos. El primero consiste en emplear un modelo predictivo que nos permita generar estimaciones numéricas precisas de la variable objetivo. El segundo objetivo implica el desarrollo de un modelo de clasificación que nos permitirá segmentar los datos por medio de  categorías que representan diferentes niveles de alerta relacionados con la prevalencia de trastornos depresivos por cada 100,000 habitantes. Para abordar este análisis, se emplean dos algoritmos de aprendizaje automático: redes neuronales y máquinas de vectores de soporte (SVM, por sus siglas en inglés). Estos modelos han sido seleccionados debido a su capacidad para manejar conjuntos de datos complejos y de alta dimensión, así como su conveniencia para la predicción y evaluación de enfermedades mentales.

## Análisis de Datos

### Características y Procesamiento de los Datos

El conjunto de datos *Mental Health Depression Disorder Data* incluye variables relacionadas con los desórdenes de depresión, organizadas en seis hojas de un documento de Excel. Las hojas contienen información sobre el uso de sustancias, nivel de educación, edad, género, conteo total por país de la prevalencia de la depresión y tasas de suicidio. 

Después de una rigurosa exploración de datos la cual consistió en analizar cada una de las hojas del documento Excel, se realizó el siguiente proceso para cada una, el cual se presenta detallado en el [notebook de exploración](https://github.com/markalbrand56/MD-Proyecto/blob/main/proyecto-g6.ipynb):

- 1. Verifación de valores nulos
- 2. Generación de ProfileReport, en donde se analizó cada una de las features para esa hoja.
- 3. Limpieza de datos, en donde se optó por eliminar los datos faltantes, por diferentes razones explicadas en el [notebook de exploración](https://github.com/markalbrand56/MD-Proyecto/blob/main/proyecto-g6.ipynb)
- 4. Separación de variables cuantitativas y cualitativas, en donde para cada una se analizó el tipo de dato, se analizó la dispersión de los datos por medio de diagramas de caja y bigotes y se generó un histrograma para representar la distribución de cada una.
- 5. Se analizó la correlación para cada una de las variables. Para determinar que tan relacionadas se encontraban entre sí.

#### Variable respuesta

Después de una rigurosa exploración de datos, se seleccionó “Depressive disorder rates (number suffering per 100,000)” como la variable objetivo. La selección de esta variable, se basó en la recomendación de Caitlin Bryson, estudiante de psicología y practicante en Conway Regional Rehabilitation Hospital. Caitlin sugiere que sería útil comparar las diferentes tasas de depresión entre varios países y también examinar las tasas de intentos de suicidio o la prevalencia de comportamientos autolesivos. Este enfoque podría revelar los peligros del empeoramiento de la depresión. Por ejemplo, si dos países tienen una tasa de depresión del 20% y una alta tasa de suicidio, esta información puede ser crucial para identificar umbrales adecuados y mejores prácticas para la intervención. Basándose en esta perspectiva, la variable objetivo permite no solo medir el alcance del problema sino también identificar patrones y tendencias que podrían indicar áreas de mayor riesgo y necesidad de intervención.

Además, del conjunto total de variables se seleccionaron las que, a través del análisis exploratorio de los datos, se consideraron adecuadas, resultando en un conjunto de datos de veinticuatro variables. Se decidió esto con la finalidad de poder analizar la presencia de desórdenes depresivos por país y su evolución a través del tiempo, tomando en cuenta distintas métricas de desórdenes mentales y la prevalencia general de trastornos depresivos.

A partir de esto, se realizó una matriz de correlación con la variables elegidas que puede ser encontrada en el notebook [correlation.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/correlation.ipynb). Por último, se exportó un archivo CSV con las variables seleccionadas.

#### Método utilizado para obtener los conjuntos de entrenamiento y prueba

Para el procesamiento de los datos, se realizaron procesos de limpieza, escalado y optimización de parámetros. Los datos hasta 2016 se utilizaron para entrenamiento (80%) y prueba (20%), y los datos de 2017 en adelante para calculo de predicciones.

> El análisis de los datos atípicos se realizó en el notebook [proyecto-g6.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/proyecto-g6.ipynb) y el análisis de clases desbalanceadas en el notebook [SVM.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/SVM.ipynb)

Analizando la variable objetivo mediante el diagrama de caja y bigotes, se observó la presencia de datos atípicos, sobre todo por encima de la media. Se decidió no eliminarlos, ya que estos datos podrían ser importantes para el análisis y la predicción. El rango de la variable es de 2065.4519 a 6096.4376, con una media de 3332.6499.

El análisis de la variable objetivo desde el enfoque de clases mostró la siguiente distribución:

Depression_binary

| Clase | Cantidad de datos     |
| ----- | --------------------- |
|0      |  1192                 |
|1      |  3011                 |
|2      |  1089                 |

Donde 0 es una baja tasa de depresión, 1 es una tasa media y 2 es una tasa alta. Se observó que la clase 1 es la más común, seguida por la clase 2 y la clase 0. Se decidió no realizar un balanceo de clases, ya que las diferencias no eran significativas y se consideró que el modelo podría manejarlas adecuadamente.

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

El ajuste de la red neuronal consistió en la prueba de diferentes configuraciones de capas. La primera configuración fue la más simple:

``` python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

```

Esta configuración mostró un rendimiento aceptable, más no excelente. El error absoluto medio (MAE) y el error cuadrático medio (MSE) durante las fases de entrenamiento y evaluación superaban los 120 y 1200 respectivamente.

Luego de una investigación y la prueba de distintas configuraciones sugeridas, se encontró que la siguiente configuración mostraba un mejor rendimiento:

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

### Aplicación de los algoritmos seleccionados en los avances al conjunto de datos de entrenamiento.

La aplicación de los algoritmos se puede observar tanto en el notebook [red_neuronal.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/red_neuronal.ipynb) para la red neuronal como en el notebook [svm.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/SVM.ipynb) para la máquina de vectores de soporte.

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

| Fase                      | Precision | Recall | F1-score | Accuracy |
|---------------------------|-----------|--------|----------|----------|
| Entrenamiento y prueba    | 0.989     | 0.99   | 0.99     | 0.99     |
| Predicciones              | 0.994     | 0.99   | 0.99     | 0.99     |

Como se observa, los resultados tanto en la fase de entrenamiento como de validación son bastante altos, lo que indica que los modelos son altamente eficaces y confiables para la detección de desórdenes de depresión en poblaciones amplias. 

En el notebook [svm.ipynb](https://github.com/markalbrand56/MD-Proyecto/blob/main/SVM.ipynb) se encuentran las matrices de confusión tanto para el entrenamiento como para la predicción del modelo.

## Selección del mejor modelo. Revisión del ajuste del modelo.

El modelo elegido es el SVM debido a su eficiencia. Podemos observar métricas bastante buenas   tanto en el entrenamiento como en las predicciones, esto demuestra la capacidad del modelo de clasificar los datos en las clases debidas.

## Conclusiones

- Los clusters identificados por el algoritmo K-means mostraron patrones y características claras en los datos.
- La red neuronal y el SVM mostraron buenos resultados en la predicción de la depresión,siendo capaces tanto de clasificar los datos con alta precisión como de predecir el número exacto de porcentaje de depresión cada 100,000 habitantes.

## Referencias

- Suriano, A. (2024). Naive Bayes & Laplace smoothing, SVM, Arboles de Decision. Departamento de Ciencias de la Computación y Tecnologías de la Información, Universidad del Valle de Guatemala.
- Carrillo García, S. (2019). Artículo científico. En S. Carrillo García, L. M. Toro Calderón, A. X. Cáceres González y E. C. Jiménez Lizarazo, Caja de herramientas. Géneros Textuales. Universidad Santo Tomás.
- CRAI USTA Bucaramanga. (2020). Informe de recursos y servicios bibliográficos. Universidad Santo Tomás.
- Vargas Leal, V. M., Galvis García, R. E., Idárraga Ortiz, S. A. y López Báez, J. D. (2020). Guía resumen del estilo APA: séptima edición. Universidad Santo Tomás. http://hdl.handle.net/11634/34384
- IBM. (2021, 17 agosto). El modelo de redes neuronales. https://www.ibm.com/docs/es/spss-modeler/saas?topic=networks-neural-model
- Del Cid, M. T. C. (2021). La depresión y su impacto en la salud pública. Revista médica hondureña, 89(Supl. 1), 46-52.
- Muñoz, V., Alvarado, C. L. A., Barros, J. M. T., & Malla, M. I. M. (2021). Prevalencia de depresión y factores asociados en adolescentes: Artículo original. Revista Ecuatoriana de Pediatría, 22(1), 6-1.
- Organización Mundial de la Salud (2023, Marzo 31). Depresión. Organización Mundial de la Salud. https://www.who.int/es/news-room/fact-sheets/detail/depression#:~:text=Se%20estima%20que%20el%203,personas%20sufren%20depresi%C3%B3n%20(1).
