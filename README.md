# Explorando los Determinantes de la Depresión: Un Análisis Global

**Karen Jimena Hernández Ortega**  
**Mark Alexander Albrand Mendoza**  
**Mario Cristales**  
**Javier Heredia**  
**Luis Roberto Furlan Collver**  
Departamento de Ciencias de la Computación. Facultad de Ingeniería, Universidad del Valle de Guatemala

## Resumen 
Este estudio analiza los factores asociados a la depresión en una perspectiva global utilizando el conjunto de datos *Mental Health Depression Disorder Data*. Se centra en la variable *Depressive disorder rates (number suffering per 100,000)* como objetivo principal, aplicando algoritmos de aprendizaje automático, específicamente redes neuronales y máquinas de vectores de soporte (SVM). Los resultados indican patrones significativos en la prevalencia de la depresión asociados a variables sociodemográficas. 

**Palabras clave**: depresión, análisis de datos, aprendizaje automático, redes neuronales, SVM.

## Introducción 
La depresión es un trastorno mental complejo influenciado por factores sociales, psicosociales, genéticos y biológicos. Se caracteriza por síntomas como la pérdida de interés en actividades cotidianas, baja autoestima y una sensación persistente de tristeza (Muñoz et al., 2021). La Organización Mundial de la Salud (OMS) estima que el 3,8% de la población mundial sufre de depresión, afectando a 280 millones de personas globalmente. Este estudio utiliza el conjunto de datos *Mental Health Depression Disorder Data* para investigar los factores asociados a la depresión en diferentes países y grupos demográficos.

## Metodología de Análisis y Recolección de Datos 
### Características y Procesamiento de los Datos
El conjunto de datos *Mental Health Depression Disorder Data* incluye variables relacionadas con los desórdenes de depresión, organizadas en seis hojas de un documento de Excel. Las hojas contienen información sobre el uso de sustancias, nivel de educación, edad, género, conteo total por país de la prevalencia de la depresión y tasas de suicidio.

Después de una rigurosa exploración de datos, se seleccionó “Depressive disorder rates (number suffering per 100,000)” como la variable objetivo. Además, se eligieron 24 variables adicionales que se consideraron relevantes para el análisis.

Para el procesamiento de los datos, se realizaron procesos de limpieza, escalado y optimización de parámetros. Los datos hasta 2016 se utilizaron para entrenamiento (80%) y prueba (20%), y los datos de 2017 en adelante para validación.

### Algoritmos Utilizados
#### Red Neuronal
Las redes neuronales, inspiradas en el funcionamiento del sistema nervioso humano, son adecuadas para manejar conjuntos de datos complejos y de alta dimensión. La red neuronal utilizada en este estudio tenía siete capas, incluyendo capas convolucionales, de max pooling, aplanado y densas. Se entrenó con 20,000 iteraciones usando un equipo con un procesador Ryzen 9 7940HS y 32 GB de RAM.

#### Máquina de Vectores de Soporte (SVM)
El SVM es un algoritmo supervisado que busca encontrar el hiperplano óptimo para clasificar datos en un espacio N-dimensional. Se utilizó un kernel lineal ya que los datos eran linealmente separables. Se creó una nueva variable *Depression_binary* para clasificar las tasas de depresión en tres categorías: bajo, medio y alto.

## Resultados
### Exploración de los Datos
Se realizó un análisis exploratorio de los datos para identificar patrones y características. Se encontró que hay un total de 196 países diferentes con datos desde 1990 hasta 2017, distribuidos uniformemente.

### Predicción y Aplicación de Algoritmos
Se utilizó el algoritmo de agrupamiento K-means para definir diez grupos óptimos basados en la suma total de las distancias cuadráticas dentro de los grupos (WCSS). Los clusters creados se presentan en la siguiente tabla:

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

Los grupos más notables fueron el primero, el tercero y el décimo. El primer grupo tenía las menores tasas de esquizofrenia, desórdenes alimenticios y uso de drogas. El tercer grupo tenía las menores tasas en variables de edad y género. El décimo grupo tenía el promedio más bajo en desórdenes bipolares.

#### Red Neuronal
La red neuronal mostró un error absoluto medio (MAE) y un error cuadrático medio (MSE) durante las fases de entrenamiento y evaluación, indicando un buen ajuste a los datos. Sin embargo, los datos de validación mostraron errores más elevados, lo cual era esperado debido a la falta de exposición del modelo a estos datos durante el entrenamiento.

| Fase          | MAE    | MSE          |
|---------------|--------|--------------|
| Entrenamiento | 2.3356 | 10.0856      |
| Evaluación    | 2.3274 | 9.9416       |
| Validación    | 468.5533 | 351,443.73489 |

#### SVM
El SVM mostró un rendimiento excepcional en la clasificación de los datos, con altas métricas de precisión, recall, F1-score y accuracy.

| Fase          | Precision | Recall | F1-score | Accuracy |
|---------------|-----------|--------|----------|----------|
| Entrenamiento | 0.989     | 0.99   | 0.99     | 0.99     |
| Validación    | 0.994     | 0.99   | 0.99     | 0.99     |

## Conclusiones
Los datos tienen una capacidad limitada para ser separados en clusters claros. La red neuronal y el SVM mostraron buenos resultados en la predicción de la depresión. La edad mostró una alta correlación con la depresión, lo que es significativo para predicciones futuras.

## Referencias
- Suriano, A. (2024). Naive Bayes & Laplace smoothing, SVM, Arboles de Decision. Departamento de Ciencias de la Computación y Tecnologías de la Información, Universidad del Valle de Guatemala.
- Carrillo García, S. (2019). Artículo científico. En S. Carrillo García, L. M. Toro Calderón, A. X. Cáceres González y E. C. Jiménez Lizarazo, Caja de herramientas. Géneros Textuales. Universidad Santo Tomás.
- CRAI USTA Bucaramanga. (2020). Informe de recursos y servicios bibliográficos. Universidad Santo Tomás.
- Vargas Leal, V. M., Galvis García, R. E., Idárraga Ortiz, S. A. y López Báez, J. D. (2020). Guía resumen del estilo APA: séptima edición. Universidad Santo Tomás. http://hdl.handle.net/11634/34384
- IBM. (2021, 17 agosto). El modelo de redes neuronales. https://www.ibm.com/docs/es/spss-modeler/saas?topic=networks-neural-model
- Del Cid, M. T. C. (2021). La depresión y su impacto en la salud pública. Revista médica hondureña, 89(Supl. 1), 46-52.
- Muñoz, V., Alvarado, C. L. A., Barros, J. M. T., & Malla, M. I. M. (2021). Prevalencia de depresión y factores asociados en adolescentes: Artículo original. Revista Ecuatoriana de Pediatría, 22(1), 6-1.
- Organización Mundial de la Salud (2023, Marzo 31). Depresión. Organización Mundial de la Salud. https://www.who.int/es/news-room/fact-sheets/detail/depression#:~:text=Se%20estima%20que%20el%203,personas%20sufren%20depresi%C3%B3n%20(1).
