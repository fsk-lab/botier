import pytest

import torch
from botier import AuxiliaryObjective


def test_auxiliary_objective_initialization():
    with pytest.raises(ValueError):
        # Should raise ValueError because neither calculation function nor output index is provided
        AuxiliaryObjective()

    with pytest.raises(ValueError):
        # Should raise ValueError because upper bound is less than lower bound
        AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=1.0, abs_threshold=0.5)

    with pytest.raises(ValueError):
        # Should raise ValueError because neither abs_threshold nor rel_threshold is provided
        AuxiliaryObjective(maximize=True, upper_bound=1.0, lower_bound=0.0)

    with pytest.raises(ValueError):
        # Should raise ValueError because rel_threshold is not in [0, 1]
        AuxiliaryObjective(maximize=True, upper_bound=1.0, lower_bound=0.0, rel_threshold=1.5)

    aux_obj = AuxiliaryObjective(maximize=True,  calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0, abs_threshold=0.5)
    assert aux_obj.abs_threshold == 0.5
    assert aux_obj.best_value == 1.0
    assert aux_obj.worst_value == 0.0


def test_auxiliary_objective_forward():
    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    X = torch.tensor([1.0, 2.0])
    aux_obj = AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, abs_threshold=2.0)
    result = aux_obj(Y, X)
    expected = torch.tensor([2.0, 5.0])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    aux_obj = AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=4.0, lower_bound=0.0, rel_threshold=0.5)
    result = aux_obj(Y, X, normalize=True)
    expected = torch.tensor([1.0, 2.0])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_auxiliary_objective_bounds():
    aux_obj = AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0, abs_threshold=0.5)
    assert aux_obj.bounds == (0.0, 1.0)


def test_auxiliary_objective_threshold():
    aux_obj = AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0, abs_threshold=0.5)
    assert aux_obj.threshold == 0.5
