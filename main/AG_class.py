import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class AG:
    def __init__(self, datos_train, datos_test, seed=123, nInd=50, maxIter=100):
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.nInd = nInd
        self.maxIter = maxIter
        np.random.seed(self.seed)
        
        self.X_train, self.y_train = self.load_data(self.datos_train)
        self.X_test, self.y_test = self.load_data(self.datos_test)
        self.n_features = self.X_train.shape[1]  # Número de características (atributos)
        self.ind_size = self.n_features * 2 + 1  # Tamaño del individuo: coeficientes, exponentes y constante
        
    def load_data(self, filename):
        data = pd.read_csv(filename)
        print(data.describe())  # Imprimir estadísticas descriptivas de los datos
        X = data.drop('y', axis=1).values  # Cambiar 'target' por 'y'
        y = data['y'].values  # Cambiar 'target' por 'y'
        return X, y
        
    def fitness(self, individuo, X, y):
        n_features = X.shape[1]
        coef = individuo[:n_features]
        exponents = individuo[n_features:-1]
        constant = individuo[-1]
        
        # Evitar valores negativos elevados a exponentes no enteros añadiendo una pequeña constante
        X_transformed = np.abs(X + 1e-10) ** exponents
        
        y_pred = np.sum(coef * X_transformed, axis=1) + constant
        
        return mean_squared_error(y, y_pred)
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, individuo, mutation_rate=0.07):
        for i in range(len(individuo)):
            if np.random.rand() < mutation_rate:
                individuo[i] += np.random.randn()
                

                
        return individuo

    def initialize_population(self, pop_size, ind_size):
        # Limitar valores iniciales
        return [np.random.randn(ind_size) for _ in range(pop_size)]

    def tournament_selection(self, population, fitnesses, k=3):
        selected = np.random.choice(len(population), k, replace=False)
        best = selected[np.argmin([fitnesses[i] for i in selected])]
        return population[best]

    def run(self):
        population = self.initialize_population(self.nInd, self.ind_size)
        best_fitness = float('inf')
        best_individuo = None

        for generation in range(self.maxIter):
            fitnesses = [self.fitness(individuo, self.X_train, self.y_train) for individuo in population]

            new_population = []
            for _ in range(self.nInd // 2):
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            population = new_population

            current_best_fitness = min(fitnesses)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individuo = population[np.argmin(fitnesses)]

            print(f"Generation {generation}: Best Fitness = {best_fitness}")

        n_features = self.X_test.shape[1]
        coef = best_individuo[:n_features]
        exponents = best_individuo[n_features:-1]
        constant = best_individuo[-1]
        
        # Evitar valores negativos elevados a exponentes no enteros añadiendo una pequeña constante
        X_transformed = np.abs(self.X_test + 1e-10) ** exponents
        
        y_pred = np.sum(coef * X_transformed, axis=1) + constant
        
        return best_individuo, y_pred















