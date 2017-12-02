import collections
import numpy as np
from nose.tools import ok_, eq_
from evolution_chamber.evolution_chamber import EvolutionChamber, EvolveResult


class Model:
    def __init__(self):
        self.X_ = None
        self.y_ = None
        self.score_ = None

    def fit(self, X, Y):
        self.X_ = X
        self.y_ = y

    def score(self, x_test, y_test):
        self.score_ = 0
        return self.score_

    def make_child(self):
        return type(self)()


class Winner(Model):
    def score(self, x_test, y_test):
        self.score_ = 100
        return self.score_


class SecondGenWinner(Model):
    def make_child(self):
        return Winner()


default_initial_individuals = (Model() for x in range(0, 10))
X = np.array([[i] for i in range(0, 100)])
y = np.array([[i] for i in range(0, 100)])


def test_evolution_chamber_init():
    evo_chamber = EvolutionChamber(X=X, y=y)
    ok_(isinstance(evo_chamber, EvolutionChamber))


def test_evolve_should_return_iterable():
    evo_chamber = EvolutionChamber(X=X, y=y)
    evolve_iterator = evo_chamber.evolve(
        default_initial_individuals,
        iteratons=10,
        limit_individuals=10
    )
    ok_(isinstance(evolve_iterator, collections.Iterable))


def test_evolve_should_return_evolve_result():
    evo_chamber = EvolutionChamber(X=X, y=y)
    evolve_iterator = evo_chamber.evolve(
        default_initial_individuals,
        iteratons=10,
        limit_individuals=10
    )
    evolve_result = next(evolve_iterator)
    ok_(isinstance(evolve_result, EvolveResult))


def test_evolve_model_fit():
    evo_chamber = EvolutionChamber(X=X, y=y)
    evolution_iter = evo_chamber.evolve(
        [Model()],
        iteratons=10,
        limit_individuals=10
    )
    evolution_result = next(evolution_iter)
    individual = evolution_result.population[0]
    X_values = np.array(individual.X_).flatten()
    are_all_values_from_first_batch = (X_values < 10).all()
    ok_(are_all_values_from_first_batch, 'model should be fitted with the first batch')


def test_evolve_winner_on_first_iteration():
    evo_chamber = EvolutionChamber(X=X, y=y)
    evolution_iter = evo_chamber.evolve(
        [Winner(), Model()],
        iteratons=10,
        limit_individuals=10
    )
    evolution_result = next(evolution_iter)
    winner = evolution_result.population[0]
    eq_(winner.score_, 100)


def test_evolve_winner_on_second_iteration():
    evo_chamber = EvolutionChamber(X=X, y=y)
    evolution_iter = evo_chamber.evolve(
        [Model(), SecondGenWinner()],
        iteratons=10,
        limit_individuals=10
    )
    first_evolution_result = next(evolution_iter)
    first_winner = first_evolution_result.population[0]
    eq_(first_winner.score_, 0)
    second_evolution_result = next(evolution_iter)
    second_winner = second_evolution_result.population[0]
    eq_(second_winner.score_, 100)


def test_evolve_survivors():
    evo_chamber = EvolutionChamber(X=X, y=y)
    evolution_iter = evo_chamber.evolve(
        [Winner(), Winner(), Model()],
        iteratons=10,
        limit_individuals=2
    )
    next(evolution_iter)
    second_result = next(evolution_iter)
    eq_(len(second_result.survivors), 2)
    eq_(second_result.survivors[0].score_, 100)
    eq_(second_result.survivors[1].score_, 100)
