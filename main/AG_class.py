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
        self.n_features = self.X_train.shape[1] *2 + 1 
        
    def load_data(self, filename):
        data = pd.read_csv(filename)
        print(data.describe())  # imprime una estadistica de los datos al inicio
        X = data.drop('y', axis=1).values
        y = data['y'].values  
        return X, y
        
    def fitness(self, individuo, X, y):
        n_features = X.shape[1]
        X_transformed = (X+ 1e-10) ** np.round(individuo[n_features:-1]).astype(int)
        
        y_pred = np.sum(individuo[:n_features] * X_transformed, axis=1) + individuo[-1]
        return mean_squared_error(y, y_pred)
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, individuo, mutation_rate=0.05):
        for i in range(len(individuo)):
            if np.random.rand() < mutation_rate:
                individuo[i] += np.random.randn()
        return individuo

    def initialize_population(self, pop_size, n_features):
        return [np.random.randn(n_features) for _ in range(pop_size)]

    def tournament_selection(self, population, fitnesses, k=3):
        selected = np.random.choice(len(population), k, replace=False)
        best = selected[np.argmin([fitnesses[i] for i in selected])]
        return population[best]

    def run(self):
        population = self.initialize_population(self.nInd, self.n_features)
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
            X_transformed = (self.X_test + 1e-10) ** np.round(best_individuo[n_features:-1]).astype(int)

            y_pred = np.sum(best_individuo[:n_features] * X_transformed, axis=1) + best_individuo[-1]
        return best_individuo, y_pred















