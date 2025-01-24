from typing import Tuple, Type, Union
import inspect
import warnings

# Torch & GPyTorch imports
import torch
from torch import Tensor
torch.set_default_dtype(torch.float64)
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood

from scipy.optimize import differential_evolution

# Botorch imports
from botorch.acquisition.monte_carlo import MCAcquisitionObjective, MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import MCMultiOutputObjective
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy, gen_candidates_torch
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning


def generate_seed_data(surface: MultiObjectiveTestProblem, budget: int, random_seed: int = 42) -> Tuple[Tensor, Tensor]:
    """
    Generates seed data for the optimization campaign by sampling `budget` points from the input space using a Sobol
    sequence.

    Args:
        surface: The multi-objective test problem to be optimized.
        budget: The number of seed points to generate.
        random_seed: The random seed to use for the Sobol sequence.

    Returns:
        x: A tensor of the seed input data (shape: `budget x dim`).
        y: A tensor of the seed output data (shape: `budget x n_objectives`).
    """
    sampler = SobolEngine(dimension=surface.dim, scramble=True, seed=random_seed)
    x = sampler.draw(budget, dtype=torch.get_default_dtype()) * (surface.bounds[1] - surface.bounds[0]) + surface.bounds[0]
    y = surface(x)

    return x, y


def _train_surrogate(x: Tensor, y: Tensor, bounds: Tensor) -> SingleTaskGP:
    """
    Internal utility that trains a botorch SingleTaskGP surrogate model on the passed x-y data. Includes a fallback for
    the cases in which MLL fitting is unsuccessful.

    Args:
        x: Tensor (shape: `n x dim`) of inputs.
        y: Tensor (shape: `n x n_outputs`) of outputs.
        bounds: Tensor (shape: `2 x dim`) of input space bounds (required for input normalization).

    Returns:
        Model: A trained botorch single-task GP model.

    Raises:
        RuntimeError: If the surrogate model cannot be fitted after multiple attempts.
    """
    if len(y.shape) == 1:
        y = y.unsqueeze(-1)

    surrogate = SingleTaskGP(
        x,
        y,
        input_transform=Normalize(d=x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)

    for algorithm in [
        (fit_gpytorch_mll_scipy, {"method": "L-BFGS-B"}),
        (fit_gpytorch_mll_torch, {})
    ]:
        try:
            fit_gpytorch_mll(mll, optimizer=algorithm[0], optimizer_kwargs=algorithm[1])
            return surrogate
        except (RuntimeError, ModelFittingError):
            continue

    raise RuntimeError("Could not fit the surrogate model after multiple attempts.")


def _train_joint_surrogate(x: Tensor, y: Tensor, bounds: Tensor) -> KroneckerMultiTaskGP:
    """
    Internal utility that trains a botorch KroneckerMultiTaskGP surrogate model on the passed x-y data. Includes a
    fallback for the cases in which MLL fitting is unsuccessful.

    Args:
        x: Tensor (shape: `n x dim`) of inputs.
        y: Tensor (shape: `n x n_outputs`) of outputs.
        bounds: Tensor (shape: `2 x dim`) of input space bounds (required for input normalization).

    Returns:
        Model: A trained botorch joint GP model.

    Raises:
        ValueError: If the input data is single-output.
        RuntimeError: If the surrogate model cannot be fitted after multiple attempts.
    """
    if len(y.shape) == 1 or y.shape[-1] == 1:
        raise ValueError("Cannot train a joint surrogate model on a single-output dataset.")

    x_scaled = (x - bounds[0]) / (bounds[1] - bounds[0])  # KroneckerMultiTaskGP + input_transform doesn't work properly

    surrogate = KroneckerMultiTaskGP(x_scaled, y, outcome_transform=Standardize(m=y.shape[-1]))
    mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)

    for algorithm in [
        (fit_gpytorch_mll_scipy, {"method": "L-BFGS-B"}),
        (fit_gpytorch_mll_torch, {})
    ]:
        try:
            fit_gpytorch_mll(mll, optimizer=algorithm[0], optimizer_kwargs=algorithm[1])
            return surrogate
        except (RuntimeError, ModelFittingError):
            continue

    raise RuntimeError("Could not fit the surrogate model after multiple attempts.")


def _setup_acqf_kwargs(obj_values: Tensor, acqf_type: Type[MCAcquisitionFunction], acqf_kwargs: dict) -> dict:
    """
    Internal utility to set up the acquisition function keyword arguments, based on the passed acquisition function.
        - For qExpectedImprovement and qProbabilityOfImprovement (i.e. for those acqfs that require a `best_f`
          argument), the `best_f` value is set to the maximum value of the objective values.
        - For qExpectedHypervolumeImprovement (i.e. for those that require a `partitioning` argument), the
          reference point is set to the value passed in `acqf_kwargs` (or to the origin as default), and a
          FastNondominatedPartitioning object is created.

    Args:
        obj_values: Tensor (shape: `n_points x n_objectives`) of objective values.
        acqf_type: The type of acquisition function to be used.
        acqf_kwargs: The keyword arguments to be passed to the acquisition function.

    Returns:
        dict: The updated acquisition function keyword arguments.
    """
    if "best_f" in inspect.signature(acqf_type.__init__).parameters:
        acqf_kwargs["best_f"] = torch.max(obj_values.flatten())

    if "partitioning" in inspect.signature(acqf_type.__init__).parameters:
        ref_point = acqf_kwargs.pop("ref_point", torch.zeros(obj_values.shape[1]))
        acqf_kwargs["partitioning"] = FastNondominatedPartitioning(Y=obj_values, ref_point=ref_point)
        acqf_kwargs["ref_point"] = ref_point

    return acqf_kwargs


def _optimize_botorch_acqf(acqf: MCAcquisitionFunction, bounds: torch.Tensor, manual_scaling: bool = False) -> Tensor:
    """
    Internal utility that optimizes the acquisition function over the input space. Contains a fallback mechanism in case
    the default optimization methods fail.

    Args:
        acqf: The acquisition function to be optimized.
        bounds: The bounds of the input space.
        manual_scaling: If True, the input space is manually scaled to [0, 1] before optimization. This is necessary for
                        the KroneckerMultiTaskGP, where the automated InputTransform does weird things.

    Returns:
        Tensor: The recommended new point (shape: `1 x dim`).

    Raises:
        RuntimeError: If the optimization fails after multiple attempts.
    """
    # perform manual scaling if necessary
    if manual_scaling is True:
        bounds_scaled = torch.stack([torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])], dim=0)
    else:
        bounds_scaled = bounds

    # optimize the acqf with some different fallback methods in case the first one fails
    optim_kwargs = {"bounds": bounds_scaled, "q": 1, "num_restarts": 10, "raw_samples": 1024}

    def differential_evolution_fallback(fun, x0, args, jac, hess, hessp, bounds, constraints, callback, **kwargs):
        """Wrapper for scipy's differential evolution optimizer as a fallback in case the gradient optimizers fail."""
        return differential_evolution(func=fun, bounds=bounds, constraints=constraints, x0=x0, args=args,
                                      callback=callback, **kwargs)

    optimizers = [
        {"gen_candidates": gen_candidates_scipy, "options": {"method": "L-BFGS-B"}},
        {"gen_candidates": gen_candidates_torch, "options": {}},
        {"gen_candidates": gen_candidates_scipy, "options": {"method": differential_evolution_fallback, "with_grad": False}},
    ]

    for optimizer in optimizers:
        try:
            candidates, _ = optimize_acqf(acqf, **optimizer, **optim_kwargs)

            if manual_scaling is True:
                candidates = candidates * (bounds[1] - bounds[0]) + bounds[0]

            return candidates.detach()

        except RuntimeError:
            continue

    raise RuntimeError("Could not optimize the acquisition function after multiple attempts.")


def recommend_new_point(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        acqf_type: Type[MCAcquisitionFunction],
        acqf_kwargs: dict,
        acq_objective: Union[MCAcquisitionObjective, MCMultiOutputObjective],
        bounds: torch.Tensor,
        joint_model: bool = False,
        black_box_objective: bool = False,
        num_posterior_samples: int = 512,
        random_seed: int = 42
) -> torch.Tensor:
    """
    Wrapper function that recommends a new point by running a full BO loop:
        1) Fit a surrogate model (using the passed x-y data),
        2) Optimize the acquisition function over the passed objective.

    If the optimization fails for any reason (either during model fitting or acquisition function optimization), a
    random point is returned instead.

    Args:
        x_train: Tensor (shape: `n x dim`) of input training data.
        y_train: Tensor (shape: `n x n_outputs`) of output training data.
        acqf_type: The type of acquisition function to be used.
        acqf_kwargs: Dictionary of keyword arguments to be passed to the acquisition function.
        acq_objective: The acquisition objective to be used.
        bounds: The bounds of the input space (shape: `2 x dim`).
        joint_model: If True, a joint GP model is trained on the input-output data.
        black_box_objective: If True, uses a black-box objective (rather than a composite one) for the acquisition function.
        num_posterior_samples: The number of posterior samples to draw for the acquisition function.
        random_seed: The random seed to use for the Sobol sequence.

    Returns:
        Tensor: The recommended new point (shape: `1 x dim`).
    """
    obj_values = acq_objective(y_train, x_train)  # shape: `n_points` or `n_points x n_objectives`

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_posterior_samples]), seed=random_seed)
    acqf_kwargs = _setup_acqf_kwargs(obj_values, acqf_type, acqf_kwargs)

    try:
        # Train the surrogate model and instantiate the acquisition function
        if black_box_objective is True:
            if joint_model is True:
                surrogate = _train_joint_surrogate(x_train, obj_values, bounds)
            else:
                surrogate = _train_surrogate(x_train, obj_values, bounds)

            acqf = acqf_type(model=surrogate, sampler=sampler, **acqf_kwargs)

        else:
            if joint_model is True:
                surrogate = _train_joint_surrogate(x_train, y_train, bounds)
            else:
                surrogate = _train_surrogate(x_train, y_train, bounds)

            acqf = acqf_type(model=surrogate, sampler=sampler, objective=acq_objective, **acqf_kwargs)

        candidate = _optimize_botorch_acqf(acqf, bounds, manual_scaling=joint_model)

    except RuntimeError as e:
        warnings.warn(f"Could not recommend a new point as a result of {e}. Returning a random point instead.")
        candidate = torch.rand(1, x_train.shape[-1]) * (bounds[1] - bounds[0]) + bounds

    return candidate.detach()
