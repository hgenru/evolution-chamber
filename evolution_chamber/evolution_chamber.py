import collections
from random import random
from sklearn.model_selection import train_test_split


class EvolveResult():
    def __init__(self, population, survivors):
        self.population = population
        self.survivors = survivors


class EvolutionChamber:
    def __init__(self, X=None, y=None):
        self._X = X
        self._y = y

    def evolve(
        self,
        initial_individuals: collections.Iterable,
        iteratons: int = 100,
        limit_individuals: int = 100,
        test_size: float = 0.3,
        probability_additional_offspring: float = 0.3
    ) -> collections.Iterator:
        population = initial_individuals
        for iteration in range(1, iteratons + 1):
            X_train, X_test, y_train, y_test = self._split_data(test_size, iteration, iteratons)
            for individual in population:
                individual.fit(X_train, y_train)
                individual.score(X_test, y_test)
            current_population_by_score = sorted(population, key=lambda i: i.score_, reverse=True)
            current_survivors = current_population_by_score[0:limit_individuals]
            yield EvolveResult(current_population_by_score, current_survivors)
            next_population = []
            for survivor in current_survivors:
                next_population.append(survivor.make_child())
                if random() < probability_additional_offspring:
                    next_population.append(survivor.make_child())
            population = next_population

    def _split_data(self, test_size, current_iteration, all_iterations):
        to = (len(self._X) // all_iterations) * current_iteration
        current_X = self._X[0:to]
        current_y = self._y[0:to]
        return train_test_split(current_X, current_y, test_size=test_size)
