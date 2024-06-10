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


ag = AG(
    datos_train=nombre_dataset_train,
    datos_test=nombre_dataset_val,
    seed=123,
    nInd=100,
    maxIter=150
)


inicio = time.time()
ind, y_pred = ag.run()
fin = time.time()
print(f'Tiempo ejecucion: {(fin-inicio):.2f} segundos')


print(f'Mejor individuo: {ind}')

print(f'Predicciones: {y_pred}')


