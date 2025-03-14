from typing import Any, Type
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from botier import AuxiliaryObjective


class Emulator:
    """
    Emulates a dataset (in the format specified in Olympus) using a regression model from scikit-learn.

    Args:
        model_type: The type of regression model to use.
        model_kwargs: Keyword arguments to pass to the model constructor.
        data_file: Path to the training data file.
        bounds: Tensor of bounds for the input space (shape: `2 x n_features`).
        maximize: Whether the problem is a maximization or minimization problem.
    """

    def __init__(
            self,
            model_type: type,
            model_kwargs: dict[str, Any],
            data_file: Path,
            bounds: torch.Tensor,
            maximize: bool = False,  # whether it is a maximization or minimization problem,
    ):

        data = pd.read_csv(data_file, header=None)
        self._x = data.iloc[:, :-1].to_numpy()
        self._y = data.iloc[:, -1].to_numpy()
        self._bounds = bounds.numpy()

        if self._x.shape[1] != bounds.shape[1]:
            raise ValueError("The number of features in the data does not match the number of features in the bounds.")

        self._maximize = maximize

        self._model_type = model_type
        self._model_kwargs = model_kwargs

        self._model, self._trained = None, False

    @property
    def dim(self) -> int:
        """
        Returns the feature dimensionality of the input space.

        Returns:
            int: The number of features.
        """
        return self._x.shape[1]

    @property
    def train_size(self) -> int:
        """
        Returns the number of training samples.

        Returns:
            int: The number of training samples.
        """
        return self._x.shape[0]

    @property
    def best_value(self) -> float:
        """
        Returns the "best" value (i.e. the maximum for a maximization problem, and the minimum for a minimization
        problem) of the target function, as observed in the training data. For a random forest regressor, this is
        equivalent to the best obtainable value.

        Returns:
            float: The best value of the target function.
        """
        if self._maximize:
            return np.max(self._y)
        return np.min(self._y)

    @property
    def worst_value(self) -> float:
        """
        Returns the "worst" value (i.e. the minimum for a maximization problem, and the maximum for a minimization
        problem) of the target function, as observed in the training data. For a random forest regressor, this is
        equivalent to the worst obtainable value.
        """
        if self._maximize:
            return np.min(self._y)
        return np.max(self._y)

    @property
    def bounds(self) -> torch.Tensor:
        """
        Returns the bounds of the input space.

        Returns:
            torch.Tensor: The bounds of the input space (shape: `2 x n_features`).
        """
        return torch.tensor(self._bounds, dtype=torch.get_default_dtype())

    def _initialize_model(self):
        """
        Deletes a trained model, and instantiates a fresh one.
        """
        del self._model
        self._model = self._model_type(**self._model_kwargs)
        self._trained = False

    def train(self):
        """
        Trains the model on the full dataset.
        """
        self._initialize_model()
        self._model.fit(self._x, self._y)
        self._trained = True

    def cross_validate(self, k: int = 5) -> torch.Tensor:
        """
        Performs k-fold cross validation on the full training dataset.

        Args:
            k: The number of folds.

        Returns:
            torch.Tensor: The R^2 scores for each fold (shape: `k`).
        """
        self._initialize_model()

        scores = cross_val_score(
            estimator=self._model,
            X=self._x,
            y=self._y,
            scoring="r2",
            cv=KFold(n_splits=k, shuffle=True, random_state=42)
        )

        self._initialize_model()

        return torch.from_numpy(scores).to(torch.get_default_dtype()).flatten()

    def __call__(self, x) -> torch.Tensor:
        """
        Implements the emulator as a callable object.

        Args:
            x: Tensor of input features (shape: `n x dim`).

        Returns:
            torch.Tensor: The predicted output values (shape: `n x 1`).
        """

        if not self._trained:
            raise RuntimeError("The model has not been trained yet.")

        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()

        y = self._model.predict(x)

        return torch.from_numpy(y).to(torch.get_default_dtype()).unsqueeze(-1)


########################################################################################################################
# Train the emulators on the respective datasets, and create the `emulated_problems` dictionary.
########################################################################################################################

emulated_problems = {}

# Benzylation Impurity Prediction Dataset

benzylation = Emulator(
    model_type=RandomForestRegressor,
    model_kwargs={"random_state": 42},
    data_file=Path(__file__).parent / "emulation_data" / "benzylation_impurity.csv",
    bounds=torch.tensor([[0.2, 1.0, 0.5, 110.0], [0.4, 5.0, 1.0, 150.0]]),
    maximize=False,
)
benzylation.train()

benzylation_cost = lambda y, x: x[..., 0] * 0.455 * 31 + x[..., 0] * 0.455 * x[..., 1] * 58 + x[..., 0] * (1 + x[..., 1] + x[..., 2]) * 104
benzylation_best_cost = 0.2 * 0.455 * 31 + 0.2 * 0.455 * 1.0 * 58 + 0.2 * (1 + 1.0 + 0.1) * 104
benzylation_worst_cost = 0.4 * 0.455 * 31 + 0.4 * 0.455 * 5.0 * 58 + 0.4 * (1 + 5.0 + 1.0) * 104

emulated_problems["Benzylation"] = {
    "surface": benzylation,
    "objectives": [
        AuxiliaryObjective(  # first objective: minimize the impurity quantity
            maximize=False,
            lower_bound=benzylation.best_value,
            upper_bound=benzylation.worst_value,
            abs_threshold=4.0,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the cost
            maximize=False,
            calculation=benzylation_cost,
            upper_bound=benzylation_worst_cost,
            lower_bound=benzylation_best_cost,
            abs_threshold=100,
        ),
        AuxiliaryObjective(  # third objective: minimize the reaction temperature
            maximize=False,
            calculation=lambda y, x: x[..., 3],
            upper_bound=benzylation.bounds[1][3],
            lower_bound=benzylation.bounds[0][3],
            abs_threshold=115,
        )
    ],
}


# SNAR E-Factor Prediction Dataset
snar = Emulator(
    model_type=RandomForestRegressor,
    model_kwargs={"random_state": 42},
    data_file=Path(__file__).parent / "emulation_data" / "snar_efactor.csv",
    bounds=torch.tensor([[0.5, 1.0, 0.1, 60.0], [2.0, 5.0, 0.5, 140.0]]),
    maximize=False
)
snar.train()

snar_cost = lambda y, x: x[..., 2] / x[..., 0] * 164 + x[..., 1] * x[..., 2] / x[..., 0] * 21 + 1 / x[..., 0] * 55
snar_best_cost = 0.1 / 2.0 * 164 + 1.0 * 0.1 / 2.0 * 21 + 1 / 2.0 * 55
snar_worst_cost = 0.5 / 0.5 * 164 + 5.0 * 0.5 / 0.5 * 21 + 1 / 0.5 * 55

emulated_problems["SNAR"] = {
    "surface": snar,
    "objectives": [
        AuxiliaryObjective(  # first objective: minimize the e-factor
            maximize=False,
            lower_bound=snar.best_value,
            upper_bound=snar.worst_value,
            abs_threshold=1.0,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the cost
            maximize=False,
            calculation=snar_cost,
            upper_bound=snar_worst_cost,
            lower_bound=snar_best_cost,
            abs_threshold=50.0,
        ),
        AuxiliaryObjective(  # third objective: minimize the reaction temperature
            maximize=False,
            calculation=lambda y, x: x[..., 3],
            upper_bound=snar.bounds[1][3],
            lower_bound=snar.bounds[0][3],
            abs_threshold=100,
        )
    ],
}


# Suzuki Yield Prediction Dataset
suzuki = Emulator(
    model_type=RandomForestRegressor,
    model_kwargs={"random_state": 42},
    data_file=Path(__file__).parent / "emulation_data" / "suzuki_yield.csv",
    bounds=torch.tensor([[75.0, 0.5, 1.0, 1.5], [90.0, 5.0, 1.8, 3.0]]),
    maximize=True
)
suzuki.train()

suzuki_cost = lambda y, x: x[..., 1] * 1317 + x[..., 2] * 940 + x[..., 3] * 20
suzuki_best_cost = 0.5 * 1317 + 1.0 * 940 + 1.5 * 20
suzuki_worst_cost = 5.0 * 1317 + 1.8 * 940 + 3.0 * 20

emulated_problems["Suzuki"] = {
    "surface": suzuki,
    "objectives": [
        AuxiliaryObjective(  # first objective: maximize the yield
            maximize=True,
            lower_bound=suzuki.worst_value,
            upper_bound=suzuki.best_value,
            abs_threshold=65,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the cost
            maximize=False,
            calculation=suzuki_cost,
            upper_bound=suzuki_worst_cost,
            lower_bound=suzuki_best_cost,
            abs_threshold=3500,
        ),
        AuxiliaryObjective(  # third objective: minimize the reaction temperature
            maximize=False,
            calculation=lambda y, x: x[..., 0],
            upper_bound=suzuki.bounds[1][0],
            lower_bound=suzuki.bounds[0][0],
            abs_threshold=86,
        )
    ],
}


# Fullerene Product Quantity Prediction Dataset
fullerene = Emulator(
    model_type=RandomForestRegressor,
    model_kwargs={"random_state": 42},
    data_file=Path(__file__).parent / "emulation_data" / "fullerene_yield.csv",
    bounds=torch.tensor([[3.0, 1.5, 100.0], [31.0, 6.0, 150.0]]),
    maximize=True,
)
fullerene.train()

emulated_problems["Fullerene"] = {
    "surface": fullerene,
    "objectives": [
        AuxiliaryObjective(  # first objective: maximize the product quantity
            maximize=True,
            lower_bound=fullerene.worst_value,
            upper_bound=fullerene.best_value,
            abs_threshold=0.9,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the reagent quantity
            maximize=False,
            calculation=lambda y, x: x[..., 1],
            upper_bound=fullerene.bounds[1][1],
            lower_bound=fullerene.bounds[0][1],
            abs_threshold=1.6,
        ),
        AuxiliaryObjective(  # third objective: minimize the reaction temperature
            maximize=False,
            calculation=lambda y, x: x[..., 2],
            upper_bound=fullerene.bounds[1][2],
            lower_bound=fullerene.bounds[0][2],
            abs_threshold=105.0,
        )
    ],
}


# Silver Nanoparticle Spectral Composition Optimization Dataset
data_nanoparticles = pd.read_csv(Path(__file__).parent / "emulation_data" / "ag_nanoparticles.csv", header=None)
nanoparticles = Emulator(
    model_type=RandomForestRegressor,
    model_kwargs={"random_state": 42},
    data_file=Path(__file__).parent / "emulation_data" / "ag_nanoparticles.csv",
    bounds=torch.tensor([[4.5, 10.0, 0.5, 0.5, 200.0], [42.8, 40.0, 30.0, 20.0, 1000.0]]),
    maximize=True,
)
nanoparticles.train()

emulated_problems["Nanoparticles"] = {
    "surface": nanoparticles,
    "objectives": [
        AuxiliaryObjective(  # first objective: maximize the spectral composition
            maximize=True,
            lower_bound=nanoparticles.worst_value,
            upper_bound=nanoparticles.best_value,
            abs_threshold=0.8,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the reagent quantity
            maximize=False,
            calculation=lambda y, x: x[..., 3],
            upper_bound=nanoparticles.bounds[1][3],
            lower_bound=nanoparticles.bounds[0][3],
            abs_threshold=9.,
        )
    ],
}


# Carbon Nanotube Composite Synthesis Optimization Dataset
data_cnt = pd.read_csv(Path(__file__).parent / "emulation_data" / "cnt_composites.csv", header=None)
nanotubes = Emulator(
    model_type=KNeighborsRegressor,
    model_kwargs={"weights": "distance", "n_neighbors": 10},
    data_file=Path(__file__).parent / "emulation_data" / "cnt_composites.csv",
    bounds=torch.tensor([[15.0, 0.0, 0.0, 0.0, 0.0], [100.0, 60.0, 70.0, 85.0, 75.0]]),
    maximize=True
)
nanotubes.train()

emulated_problems["Nanotubes"] = {
    "surface": nanotubes,
    "objectives": [
        AuxiliaryObjective(  # first objective: maximize the conductivity
            maximize=True,
            lower_bound=nanotubes.worst_value,
            upper_bound=nanotubes.best_value,
            abs_threshold=350,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the reagent quantities
            maximize=False,
            calculation=lambda y, x: x[..., 1] + x[..., 2] + x[..., 3] + x[..., 4],
            upper_bound=nanotubes.bounds[1][1] + nanotubes.bounds[1][2] + nanotubes.bounds[1][3] + nanotubes.bounds[1][4],
            lower_bound=nanotubes.bounds[0][1] + nanotubes.bounds[0][2] + nanotubes.bounds[0][3] + nanotubes.bounds[0][4],
            abs_threshold=0.5
        )
    ],
}


# Enzymatic alkoxylation yield prediction dataset
data_alkoxylation = pd.read_csv(Path(__file__).parent / "emulation_data" / "alkoxylation_yield.csv", header=None)
alkoxylation = Emulator(
    model_type=KNeighborsRegressor,
    model_kwargs={"weights": "distance", "n_neighbors": 10},
    data_file=Path(__file__).parent / "emulation_data" / "alkoxylation_yield.csv",
    bounds=torch.tensor([[0.05, 0.5, 2.0, 6.0], [1.0, 10.0, 8.0, 8.0]]),
    maximize=True,
)
alkoxylation.train()

emulated_problems["Alkoxylation"] = {
    "surface": alkoxylation,
    "objectives": [
        AuxiliaryObjective(  # first objective: maximize the yield
            maximize=True,
            lower_bound=alkoxylation.worst_value,
            upper_bound=alkoxylation.best_value,
            abs_threshold=20.0,
            output_index=0
        ),
        AuxiliaryObjective(  # second objective: minimize the enzyme quantities
            maximize=False,
            calculation=lambda y, x: x[..., 0] + x[..., 1] + x[..., 2],
            upper_bound=alkoxylation.bounds[1][0] + alkoxylation.bounds[1][1] + alkoxylation.bounds[1][2],
            lower_bound=alkoxylation.bounds[0][0] + alkoxylation.bounds[0][1] + alkoxylation.bounds[0][2],
            abs_threshold=10.0
        )
    ],
}
