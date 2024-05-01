from src.utilities.constants import Verbose


class RelatednessModelBase:
    def __init__(self, verbose: Verbose = Verbose.DEFAULT):
        self.verbose = verbose
        self.name = 'Relatedness Model Base'
        self.data = AbstractDataManager

        self.model = AbstractArchitecture
        self.loss_function = abstract_loss_function
        self.optimizer = AbstractOptimizer


class AbstractArchitecture:
    pass


class AbstractDataManager:
    pass


def abstract_loss_function():
    pass


class AbstractOptimizer:
    pass
