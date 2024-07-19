import torch
from botier import HierarchyScalarizationObjective, AuxiliaryObjective


def test_hierarchy_scalarization_objective_initialization():
    # Test if the class initializes properly with a list of AuxiliaryObjective objects
    objectives = [AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., i], abs_threshold=0.5) for i in range(2)]
    hso = HierarchyScalarizationObjective(objectives=objectives)
    assert len(hso.objectives) == 2
    assert hso.scalarizer._final_obj_idx == 0
    assert hso.scalarizer._k == 1E2


def test_calculate_objective_values():
    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    X = torch.tensor([1.0, 2.0])
    objectives = [
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 0] + x, abs_threshold=2.0),
        AuxiliaryObjective(maximize=True, calculation=lambda y, x: y[..., 1] * x, abs_threshold=8.0)
    ]
    hso = HierarchyScalarizationObjective(objectives=objectives)
    result = hso.calculate_objective_values(Y, X)
    expected = torch.tensor([[2.0, 2.0], [5.0, 8.0]])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_forward():
    raise NotImplementedError()
