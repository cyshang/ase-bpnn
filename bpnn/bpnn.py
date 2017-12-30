from ase.calculators.calculator import Calculator
import tensorflow as tf

class BPNN(calculator):
    """
    """

    def __init__(self,
                 sfConfig=None, nnConfig=None,
                 atoms=None):

        self.set_sf(sfConfig)
        self.set_nn(nnConfig)

    def set_sf(self, sfConfig):
        pass

    def set_nn(self, nnConfig):
        pass

    def calculate(self, atoms):
        pass

    def train(self, dataset):
        pass

    def load_param(self, fname):
        pass

    def save_param(self, fname):
        pass
