import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from sklearn.metrics import root_mean_squared_error, r2_score
from AG_class import AG

def grid_search(ag_class, param_grid, data_folder, dataset_name):
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    dataset_train = os.path.join(data_folder, dataset_name + "_train.csv")
    dataset_val = os.path.join(data_folder, dataset_name + "_val.csv")
    results_folder = "./main/experiment/results/housing_results"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for params in param_combinations:
        print(f"Evaluating parameters: {params}")

        ag = ag_class(
            datos_train=dataset_train,
            datos_test=dataset_val,
            **params
        )

        start_time = time.time()
        ind, y_pred = ag.run()
        elapsed_time = time.time() - start_time

       
        y_true = pd.read_csv(dataset_val)['y']
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results.append((params, rmse, r2, elapsed_time, ag.fitness_history))

        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ag.fitness_history, label=f"Params: {params}")
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Convergence of Best Fitness per Generation')
        ax.legend()
        ax.grid(True)
        
        
        filename = f"fitness_plot_{params['nInd']}_{params['maxIter']}_{params['mutation_rate']}_{params['elitism_rate']}_{params['tournament_size']}_{params['crossover_rate']}.png"
        filepath = os.path.join(results_folder, filename)
        plt.savefig(filepath)
        plt.close(fig) 
        print(f"Plot saved as {filepath}")

    return results


param_grid = {
    'nInd': [50, 80, 100, 120], 
    'maxIter': [100, 150, 60 ], 
    'mutation_rate': [0.01, 0.05, 0.1],  
    'elitism_rate': [0.05, 0.1, 0.15],  
    'tournament_size': [2, 3, 5],  
    'crossover_rate': [0.6, 0.72, 0.85]  
}

data_folder = "./main/data"
dataset_name = 'housing'


results = grid_search(AG, param_grid, data_folder, dataset_name)
output_directory = './main/experiment/results_summary'
output_file_path = os.path.join(output_directory, dataset_name + '_results_summary.txt')


with open(output_file_path, 'w') as file:
   
    file.write("AG Algorithm Results Summary\n")
    file.write("----------------------------\n")
    for params, rmse, r2, time_elapsed, _ in results:
        file.write(f"Parameters: {params}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Time: {time_elapsed:.2f} seconds\n")

print(f"Results have been saved to {output_file_path}")