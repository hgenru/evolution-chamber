from nose.tools import ok_, eq_

from src.evolution_chamber import EvolutionChamber


def test_init():
    class Test:
        pass
    evo_chamber = EvolutionChamber(Test)
    ok_(isinstance(evo_chamber, EvolutionChamber))
