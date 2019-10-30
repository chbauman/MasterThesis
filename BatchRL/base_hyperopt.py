from typing import Dict, List

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample

from abc import ABC, abstractmethod

from base_dynamics_model import BaseDynamicsModel


class HyperOptimizableModel(BaseDynamicsModel, ABC):

    param_list: List[Dict] = []  #: List of tried parameters.

    @abstractmethod
    def get_space(self) -> Dict:
        """
        Defines the hyperopt space with the hyper parameters
        to be optimized for a given model.

        Returns:
            hyperopt space definition.
        """
        pass

    @abstractmethod
    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        """Configure new model with given parameters.

        Initializes another model with the parameters as
        specified by the sample, which is a sample of the specified
        hyperopt space.

        Args:
            hp_sample: Sample of hyperopt space.

        Returns:
            Another model with the same type as self, initialized
            with the parameters in the sample.
        """
        pass

    @abstractmethod
    def hyper_objective(self) -> float:
        """
        Defines the objective to be used for hyperopt.
        It will be minimized, i.e. it has to be some kind of
        loss, e.g. validation loss.
        Model assumed to be fitted first.

        Returns:
            Numerical value from evaluation of the objective.
        """
        pass

    def optimize(self, n: int = 100) -> Dict:
        """
        Does the full hyper parameter optimization with
        the given objective and space.

        Args:
            n: Number of model initializations, fits and objective
                computations.

        Returns:
            The optimized hyper parameters.
        """
        hp_space = self.get_space()
        self.param_list = []

        # Define final objective function
        def f(hp_sample: Dict) -> float:
            """Fits model and evaluates it.

            Args:
                hp_sample: Model parameters.

            Returns:
                Value of the objective.
            """

            mod = self.conf_model(hp_sample)
            self.param_list += [hp_sample]
            mod.fit()
            curr_obj = mod.hyper_objective()
            print(hp_sample, f"Objective value: {curr_obj}")
            return curr_obj

        # Do parameter search
        best = fmin(
            fn=f,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=n
        )

        return best
