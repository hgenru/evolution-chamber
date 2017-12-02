import collections
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
        test_size: int = 0.3,
        limit_individuals: int = 100
    ) -> collections.Iterator:
        individuals = initial_individuals
        for iteration in range(1, iteratons + 1):
            X_train, X_test, y_train, y_test = self._split_data(test_size, iteration, iteratons)
            for individual in individuals:
                individual.fit(X_train, y_train)
                individual.score(X_test, y_test)
            population = sorted(individuals, key=lambda i: i.score_, reverse=True)
            survivors = population[0:limit_individuals]
            individuals = [individual.make_child() for individual in survivors]
            yield EvolveResult(population, survivors)

    def _split_data(self, test_size, current_iteration, all_iterations):
        to = (len(self._X) // all_iterations) * current_iteration
        current_X = self._X[0:to]
        current_y = self._y[0:to]
        return train_test_split(current_X, current_y, test_size=test_size)

    def _next_gen(self, individuals, limit):
        pass
