import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from hyperopt import hp
from hyperopt import fmin, tpe
from hyperopt.pyll import scope as ho_scope

from base_dynamics_model import BaseDynamicsModel, construct_test_ds
from data import Dataset

OptHP = Tuple[Dict, float]  #: The type of the stored info.


def save_hp(name_hp: str, opt_hp: OptHP) -> None:
    with open(name_hp, 'wb') as f:
        pickle.dump(opt_hp, f)


def load_hp(name_hp) -> OptHP:
    with open(name_hp, 'rb') as f:
        opt_hp = pickle.load(f)
    return opt_hp


class HyperOptimizableModel(BaseDynamicsModel, ABC):
    param_list: List[Dict] = []  #: List of tried parameters.
    base_name: str  #: Base name independent of hyperparameters.
    curr_val: float = 10e100  #: Start value for optimization.

    @abstractmethod
    def get_space(self) -> Dict:
        """
        Defines the hyperopt space with the hyper parameters
        to be optimized for a given model.

        Returns:
            hyperopt space definition.
        """
        pass

    @classmethod
    @abstractmethod
    def get_base_name(cls, **kwargs):
        """Returns the unique name given all the non-hyperparameter parameters."""
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

        # Load the previously optimum if exists
        try:
            _, self.curr_val = load_hp(self.base_name)
        except FileNotFoundError:
            pass

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

            # Save if new params are better
            if curr_obj < self.curr_val:
                self.curr_val = curr_obj
                save_path = self._get_opt_hp_f_name(self.base_name)
                save_hp(save_path, (hp_sample, self.curr_val))
            return curr_obj

        # Do parameter search
        best = fmin(
            fn=f,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=n
        )

        return best

    @classmethod
    def _get_opt_hp_f_name(cls, b_name: str):
        return cls.model_path + b_name + "_OPT_HP.pkl"

    @classmethod
    def from_best_hp(cls, **kwargs):
        base_name = cls.get_base_name(**kwargs)
        name_hp = cls._get_opt_hp_f_name(base_name)
        opt_hp = load_hp(name_hp)
        hp_params, val = opt_hp
        init_params = cls._hp_sample_to_kwargs(hp_params)
        return cls(**kwargs, **init_params)

    @classmethod
    def _hp_sample_to_kwargs(cls, hp_sample: Dict):
        return hp_sample


class TestHopTable(HyperOptimizableModel):

    name: str = "TestHop"

    def __init__(self, ds: Dataset, base_param: int = 5, h_param_1: int = 0):

        super().__init__(ds, self.name)
        self.bp = base_param
        self.h_param = h_param_1

        self.base_name = self.get_base_name(base_param=base_param)

    def get_space(self) -> Dict:
        hp_space = {
            'h_param_1': ho_scope.int(hp.quniform('n_layers', low=0, high=20, q=1)),
        }
        return hp_space

    @classmethod
    def get_base_name(cls, **kwargs):
        return cls.name + "_" + str(kwargs['base_param'])

    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        new_mod = TestHopTable(self.data, self.bp, **hp_sample)
        return new_mod

    def hyper_objective(self) -> float:
        x = self.h_param
        return -x * x + 12 * x + 15

    def fit(self) -> None:
        pass

    def predict(self, in_data):
        pass


def test_hyperopt():

    ds = construct_test_ds(20)
    test_hop_mod_4 = TestHopTable(ds, 4, 6)
    test_hop_mod_4.optimize(5)

    best_mod_4 = TestHopTable.from_best_hp(ds=ds, base_param=4)
    
    print("Hyperopt test passed :)")
