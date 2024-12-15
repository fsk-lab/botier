import torch
from botier import HierarchyScalarizationObjective, AuxiliaryObjective, ObjectiveCalculator


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

    samples = torch.rand(10, 5, 3, 2)  # Simulate posterior samples with expected shape
    X = torch.tensor([1.0, 2.0])  # Simulate optional inputs

    objectives = [
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.mean(dim=-1), abs_threshold=2.0),
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.sum(dim=-1), abs_threshold=8.0)
    ]

    # Initialize the class
    hso = HierarchyScalarizationObjective(
        objectives=objectives,
        normalized_objectives=True,
    )

    # Pass through the forward function
    scalarized_output = hso.forward(samples, X)

    # Ensure that scalarized output returns a tensor with the expected shape
    # The output scalarization should reduce the dimensions to [...], depending on hierarchy scalarization
    assert scalarized_output.shape == (10, 5, 3)  # Since we aggregate over the objectives scalarization


def test_forward_not_normalized():

    samples = torch.rand(10, 5, 3, 2)  # Simulate posterior samples with expected shape
    X = torch.tensor([1.0, 2.0])  # Simulate optional inputs

    objectives = [
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.mean(dim=-1), abs_threshold=2.0),
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.sum(dim=-1), abs_threshold=8.0)
    ]

    # Initialize the class
    hso = HierarchyScalarizationObjective(
        objectives=objectives,
        normalized_objectives=False,
    )

    # Pass through the forward function
    scalarized_output = hso.forward(samples, X)

    # Ensure that scalarized output returns a tensor with the expected shape
    # The output scalarization should reduce the dimensions to [...], depending on hierarchy scalarization
    assert scalarized_output.shape == (10, 5, 3)  # Since we aggregate over the objectives scalarization


def test_objective_calculator_initialization():
    """
    Test the initialization of the ObjectiveCalculator.
    """

    objectives = [
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.mean(dim=-1), abs_threshold=2.0),
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.sum(dim=-1), abs_threshold=8.0)
    ]

    # Initialize the ObjectiveCalculator with mocked AuxiliaryObjectives
    obj_calc = ObjectiveCalculator(
        objectives=objectives,
        final_objective_idx=1,
        normalized_objectives=True,
        k=50
    )

    # Ensure the attributes are set up correctly
    assert len(obj_calc.objectives) == 2
    assert obj_calc._norm is True
    assert obj_calc.scalarizer._final_obj_idx == 1
    assert obj_calc.scalarizer._k == 50


def test_obj_calculation_forward():
    """
    Test the forward pass to ensure ObjectiveCalculator works as expected.
    """
    # Create mock input values
    samples = torch.rand(10, 5, 3, 2)  # Simulated posterior sample tensor
    X = torch.rand(10, 5, 3, 1)  # Input tensor

    objectives = [
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.mean(dim=-1), abs_threshold=2.0),
        AuxiliaryObjective(maximize=True, calculation=lambda y, x=None, normalize=True: y.sum(dim=-1), abs_threshold=8.0)
    ]

    # Initialize the ObjectiveCalculator with mocked objectives
    obj_calc = ObjectiveCalculator(
        objectives=objectives,
        normalized_objectives=True
    )

    # Call the forward pass
    result = obj_calc.forward(samples, X)

    # Ensure the output tensor has the expected shape
    # The number of objectives should match the number of mocked objectives
    assert result.shape[-1] == len(objectives)
    assert result.shape[:-1] == (10, 5, 3)  # The other dimensions should match the sample input dimensions
