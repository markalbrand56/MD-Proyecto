# Exploración de Datos del Conjunto "Mental health Depression disorder Data"

## Introducción
El dataset utilizado en este análisis es el "Mental health Depression disorder Data", obtenido de [data.world](https://data.world/vizzup/mental-health-depression-disorder-data). Este conjunto de datos contiene 54 variables distribuidas en 6 hojas diferentes, con un total de 115,044 observaciones. El objetivo del análisis es comprender la prevalencia de la depresión y sus factores asociados a nivel global.

## Variables Numéricas
Para las variables cuantitativas, se realizaron histogramas para analizar su distribución. A continuación se detallan las distribuciones encontradas en cada hoja del dataset:

- **depression-by-level-of-education**: La mayoría de las variables presentaron distribuciones uniformes, con algunas excepciones como distribuciones geométricas y binomiales.
- **suicide-rates-vs-prevalence-of-depression**: Las variables principales presentaron distribuciones geométricas, binomiales y en forma de J invertida.
- **number-with-depression-by-country**: Distribución en forma de J invertida.
- **prevalence-by-mental-and-substance**: Diversas distribuciones como binomial negativa, geométrica, log-normal y uniforme.
- **prevalence_depression_age**: Distribuciones variadas como hipergeométrica, binomial negativa, log-normal y gamma.
- **Prevelance_depression_male**: Distribución binomial y en forma de J invertida.

## Análisis de Correlación
Se analizó la correlación entre las variables cuantitativas en cada hoja. Se encontraron correlaciones significativas entre la depresión y varias variables demográficas y de salud mental.

## Variables Categóricas
Las únicas variables categóricas presentes son 'Entity' y 'Code', las cuales se analizaron mediante tablas de frecuencias.

## Variables Seleccionadas
Se determinó que las variables más relevantes para el análisis son:
- Entity
- Code
- Year
- Schizophrenia (%)
- Bipolar disorder (%)
- Eating disorders (%)
- Anxiety disorders (%)
- Drug use disorders (%)
- Depression (%)
- Alcohol use disorders (%)
- Prevalence in males (%)
- Prevalence in females (%)
- Population_x
- Suicide rate (deaths per 100,000 individuals)
- Depressive disorder rates (number suffering per 100,000)
- Population_y
- Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number)
- 20-24 years old (%)
- 10-14 years old (%)
- All ages (%)
- 70+ years old (%)
- 30-34 years old (%)
- 15-19 years old (%)
- 25-29 years old (%)
- 50-69 years old (%)
- Age-standardized (%)
- 15-49 years old (%)

## Agrupamiento (Clustering)
Se utilizó exclusivamente el algoritmo de agrupamiento K-means, que requiere definir la cantidad de grupos antes de iniciar el análisis. Para determinar el número óptimo de grupos, se examinó cómo varía la suma total de las distancias cuadráticas dentro de los grupos (WCSS) al modificar la cantidad de grupos. Como se muestra en la figura 1, seguido de diez grupos no se tiene un cambio significativo, por lo que se decidió utilizar solamente diez divisiones.

**Figura 1. Variabilidad de datos en los grupos**

El algoritmo creó diez grupos que pueden ser visualizados en la tabla 2 presentada a continuación.

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

**Tabla 2. Representación de datos por cluster**

Los grupos más notables, con la mayor cantidad de datos, fueron el primero, el tercero y el décimo. El primer grupo tiene en promedio las menores tasas de esquizofrenia, desórdenes alimenticios y uso de drogas. El tercer grupo tiene en promedio las menores tasas en las variables de edad y género. Por último, el décimo grupo se compone por datos con el promedio más bajo en la variable de desórdenes bipolares. Estos grupos permiten observar las diferencias y agrupaciones que surgen en el conjunto de datos usado.

## Modelos Predictivos
Después de analizar las agrupaciones creadas de forma no supervisada, se aplicaron varios algoritmos para realizar predicciones sobre la cantidad de personas que presentan desórdenes de depresión cada 100,000 personas. Los algoritmos seleccionados fueron una red neuronal y una máquina de vectores de soporte (SVM).

## Discusión
### Situación Problemática
Los trastornos mentales, especialmente la depresión, son una preocupación creciente a nivel global, afectando significativamente la calidad de vida y los sistemas de salud.

### Problema Científico
Se necesita una comprensión profunda de los determinantes sociodemográficos de la depresión para desarrollar estrategias de intervención efectivas.

### Objetivos
- Comprender la influencia de factores socioeconómicos, de género, demográficos y educativos en la prevalencia de la depresión.
- Identificar países y grupos de edad con mayores incrementos en la depresión.
- Analizar la relación entre tasas de suicidio y depresión.

## Conclusiones
- Los datos tienen poca capacidad de ser separados en clusters claros.
- No se encontró una asociación significativa entre el género y el porcentaje de depresión.
- La edad muestra una alta correlación con la depresión, lo que es importante para predicciones futuras.

## Bibliografía
Vizzup (2020). [Mental health Depression disorder Data](https://data.world/vizzup/mental-health-depression-disorder-data)
