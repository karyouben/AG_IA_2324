
# Proyecto AG_IA_2324

Este proyecto implementa un Algoritmo Genético (AG) para optimizar modelos de regresión utilizando diferentes conjuntos de datos. El proyecto está organizado de la siguiente manera:

## Estructura del Proyecto

![Estructura del Proyecto](https://i.ibb.co/S6VByT3/Captura-de-pantalla-2024-06-10-231136.png)




## Descripción de los Directorios y Archivos

#### main/

Directorio principal que contiene todos los scripts y datos necesarios para ejecutar el Algoritmo Genético.

#### main/data/

Contiene los conjuntos de datos utilizados en los experimentos:
- `housing_train.csv` y `housing_val.csv`: Datos de entrenamiento y validación para el conjunto de datos de housing.
- `synt1_train.csv` y `synt1_val.csv`: Datos de entrenamiento y validación para el conjunto de datos sintético 1.
- `toy1_train.csv` y `toy1_val.csv`: Datos de entrenamiento y validación para el conjunto de datos toy 1.

#### main/experiment/

Contiene los scripts de experimentos y los resultados generados:
- `__pycache__/`: Archivos caché generados por Python.
- `results/`: Directorio donde se guardan los resultados de los experimentos.
- `results_summary/`: Directorio donde se guardan los resúmenes de los resultados en formato de texto.
- `results_summary_excel/`: Directorio donde se guardan los resúmenes de los resultados en formato Excel.
- `AG_experiment_housing_parallelized.py`: Script para ejecutar el AG en el conjunto de datos de housing de forma paralelizada.
- `AG_experiment_synt1_parallelized.py`: Script para ejecutar el AG en el conjunto de datos synt1 de forma paralelizada.
- `AG_experiment_toy1_parallelized.py`: Script para ejecutar el AG en el conjunto de datos toy 1 de forma paralelizada.
- `AG_experiment_parallelized.py`: Script genérico para ejecutar el AG de forma paralelizada en diferentes conjuntos de datos. (requiere modificarlo, explicado mas adelante)

#### Scripts para ejecucion del algoritmo

- `housing_AG.py`: Script para ejecutar el AG en el conjunto de datos de housing.
- `synt1_AG.py`: Script para ejecutar el AG en el conjunto de datos synt1.
- `toy1_AG.py`: Script para ejecutar el AG en el conjunto de datos toy1.

#### Clase del Algoritmo Genético

- `AG_class.py`: Contiene la implementación de la clase AG que define el Algoritmo Genético utilizado en los experimentos.

## Uso del Proyecto

### Preparación del Entorno

1. Clonar el repositorio:
   ```sh
   git clone https://github.com/tu-usuario/AG_IA_2324.git

## Ejecución de Experimentos
### Ejecución de la busqueda en rejilla de hyperparametros de forma paralelizada

Para ejecutar el AG en el conjunto de datos de housing de forma paralelizada:

Para ejecutar `AG_experiment_parallelized.py` primero:
   `cd AG_IA_2324/main`

Para ejecutar `AG_experiment_housing_parallelized.py`, `AG_experiment_synt1_parallelized.py`, `AG_experiment_toy1_parallelized.py` primero:

   `cd AG_IA_2324`

(si no queremos pararelizarlo ejecutarlo sin meter ningún comando)

Si ejecutamos `AG_experiment_parallelized.py` de forma pararelizada debemos usar el siguiente comando en el directorio mencionado anteriormente:

   `mpiexec -n X python AG_experiment_parallelized.py`

siendo X el numero de hilos que quieras pararelizar la ejecución como por ejemplo:

   `mpiexec -n 4 python AG_experiment_parallelized.py`

para ejecutar nuevos datos modificar `dataset_name`, `dataset_val`, `dataset_train`, por ejemplo:
    
   `dataset_name = 'housing'`
   `dataset_train = os.path.join(data_folder, dataset_name + "_train.csv")`
   `dataset_val = os.path.join(data_folder, dataset_name + "_val.csv")`

para modificar los hyperparametros basta con cambiar los valores del `param_grid`, por ejemplo:

   ` param_grid = {`
        `'nInd': [100],`
        `'maxIter': [150],`
        `'mutation_rate': [0.01, 0.05, 0.1, 0.2],`
        `'elitism_rate': [0.05, 0.1, 0.15, 0.2],`
        `'tournament_size': [3, 5, 7, 9, 10, 12],`
        `'crossover_rate': [0.6, 0.72, 0.85, 0.9, 0.95]`
    `}`

 una vez ejecutado se generara una grafica por cada combinacion de hyperparametros y finalmente se generara un txt con todas las combinaciones de hyperparametros junto con su RMSE, R2 y tiempo de ejecución.

#### Resultados
Los resultados de los experimentos se guardarán en los directorios results/ y results_summary/ dentro del directorio experiment/ si se ejecuta `AG_experiment_housing_parallelized.py`, `AG_experiment_synt1_parallelized.py`, `AG_experiment_toy1_parallelized.py`

Los resultados de los experimentos se guardarán en los directorios results/ y en el main(el archivo txt) dentro del directorio main\ si se ejecuta `AG_experiment_parallelized.py`

## Ejecución del algoritmo
El algoritmo se ejecuta en los siguientes ficheros según los datos que se quieran ejecutar
- `housing_AG.py`: Script para ejecutar el AG en el conjunto de datos de housing.
- `synt1_AG.py`: Script para ejecutar el AG en el conjunto de datos synt1.
- `toy1_AG.py`: Script para ejecutar el AG en el conjunto de datos toy1.

Para ejecutarlos primero:
    `cd AG_IA_2324`

Podemos modificar dentro de estos archivos la semilla(seed), el numero de individuos(nInd) y el numero de iteraciones(maxIter):

`ag = AG(`
    `datos_train=nombre_dataset_train,`
    `datos_test=nombre_dataset_val,`
    `seed=123,`
    `nInd=100,`
    `maxIter=150`
)

para ejecutar nuevos datos modificar `nombre_dataset`, `nombre_dataset_train`, `nombre_dataset_val`, por ejemplo:

   `nombre_dataset = 'synt1'`
   `nombre_dataset_train = os.path.join(data_folder, nombre_dataset + "_train.csv")`
   `nombre_dataset_val = os.path.join(data_folder, nombre_dataset + "_val.csv")`

## Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue los pasos a continuación para contribuir:

1º Haz un fork del proyecto.
2º Crea una rama con tu nueva funcionalidad (git checkout -b feature/nueva-funcionalidad).
3º Haz commit de tus cambios (git commit -am 'Añadir nueva funcionalidad').
4º Sube tus cambios a tu fork (git push origin feature/nueva-funcionalidad).
5º Crea un nuevo Pull Request.

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Para más información, ver el archivo LICENSE.