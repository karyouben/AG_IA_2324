import os
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import time
from AG_class import AG


# Nombre generico del dataset
nombre_dataset = 'housing'

data_folder = "./main/data/"

nombre_dataset_train = os.path.join(data_folder, nombre_dataset + "_train.csv")
nombre_dataset_val = os.path.join(data_folder, nombre_dataset + "_val.csv")

# La clase AG debe estar implementada
ag = AG(
    datos_train=nombre_dataset_train,
    datos_test=nombre_dataset_val,
    seed=123,
    nInd=110,
    maxIter=150
)

# Ejecucion del AG midiendo el tiempo
inicio = time.time()
ind, y_pred = ag.run()
fin = time.time()
print(f'Tiempo ejecucion: {(fin-inicio):.2f} segundos')

# Imprimir mejor solución encontrada
print(f'Mejor individuo: {ind}')

# Imprimir predicciones sobre el conjunto de test
print(f'Predicciones: {y_pred}')

# Cargar valores reales de 'y' en el conjunto de validacion/test 
# y calcular RMSE y R2 con las predicciones del AG
y_true = pd.read_csv(nombre_dataset_val)['y']  # Cambiar 'target' por 'y'
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.4f}')

r2 = r2_score(y_true, y_pred)
print(f'R2: {r2:.4f}')

