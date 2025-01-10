import pytest

import torch
from botier import AuxiliaryObjective


def test_auxiliary_objective_initialization():
    with pytest.raises(ValueError):
        # Should raise ValueError because neither calculation function nor output index is provided
        AuxiliaryObjective()

    with pytest.raises(ValueError):
        # Should raise ValueError because neither calculation nor output index is provided
        AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=1.0, abs_threshold=0.5)

    with pytest.raises(ValueError):
        # Should raise ValueError because neither abs_threshold nor rel_threshold is provided
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0)

    with pytest.raises(ValueError):
        # Should raise ValueError because rel_threshold is provided without best and worst value
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, rel_threshold=0.80)

    with pytest.raises(ValueError):
        # Should raise ValueError because rel_threshold is not in [0, 1]
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0, rel_threshold=1.5)

    with pytest.raises(ValueError):
        # Should raise ValueError because maximize=True but abs_threshold is less than worst_value
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.75, abs_threshold=0.5)

    with pytest.raises(ValueError):
        # Should raise ValueError because maximize=True but abs_threshold is greater than best_value
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=0.90, lower_bound=0.75, abs_threshold=0.95)

    with pytest.raises(ValueError):
        # Should raise ValueError because maximize=False but abs_threshold is greater than worst_value
        AuxiliaryObjective(maximize=False, calculation=lambda y, x: y[..., 0] + x, upper_bound=0.80, lower_bound=0.75, abs_threshold=0.90)

    with pytest.raises(ValueError):
        # Should raise ValueError because maximize=False but abs_threshold is less than best_value
        AuxiliaryObjective(maximize=False, calculation=lambda y, x: y[..., 0] + x, upper_bound=0.80, lower_bound=0.75, abs_threshold=0.50)

    with pytest.raises(ValueError):
        # Should raise ValueError because upper bound is less than lower bound
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=0.0, lower_bound=1.0, abs_threshold=0.5)

    # Test with maximize=True
    aux_obj = AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0, abs_threshold=0.5)
    assert aux_obj.abs_threshold == 0.5
    assert aux_obj.best_value == 1.0
    assert aux_obj.worst_value == 0.0

    # Test with maximize=False
    aux_obj = AuxiliaryObjective(maximize=False, calculation=lambda y, x: y[..., 0] + x, upper_bound=1.0, lower_bound=0.0, abs_threshold=0.5)
    assert aux_obj.abs_threshold == 0.5
    assert aux_obj.best_value == 0.0
    assert aux_obj.worst_value == 1.0

    # Test with output_index
    aux_obj = AuxiliaryObjective(maximize=True,  calculation=None, output_index=-1, upper_bound=1.0, lower_bound=0.0, abs_threshold=0.5)
    y = torch.randn((2, 2, 10))
    assert torch.allclose(aux_obj.function(y, y), y[..., -1])


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
