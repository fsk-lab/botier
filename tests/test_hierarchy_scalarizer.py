import torch
from botier import HierarchyScalarizer
from botier.utils import smoothmin


def test_hierarchy_scalarizer_initialization():
    scalarizer = HierarchyScalarizer(final_objective_idx=1, k=1E2)
    assert scalarizer._final_obj_idx == 1
    assert scalarizer._k == 1E2


def test_hierarchy_scalarizer_forward():
    values = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    thresholds = torch.tensor([1.5, 2.5, 3.5])
    scalarizer = HierarchyScalarizer(final_objective_idx=2, k=1E2)

    result = scalarizer(values, thresholds)

    # Compute expected result using smoothmin and logistic functions
    hierarchy_masks = []
    scalarized_objectives = []

    for idx in range(values.shape[-1]):
        hierarchy_masks.append(torch.sigmoid(1E2 * (values[..., idx] - thresholds[idx])))
        smoothmin_val = smoothmin(values[..., idx], thresholds[idx], 1E2)

        if idx > 0:
            hierarchy_mask = torch.stack(hierarchy_masks[:idx], dim=-1)
            hierarchy_mask = torch.prod(hierarchy_mask, dim=-1)
            smoothmin_val = hierarchy_mask * smoothmin_val

        scalarized_objectives.append(smoothmin_val)

    hierarchy_mask = torch.stack(hierarchy_masks, dim=-1)
    hierarchy_mask = torch.prod(hierarchy_mask, dim=-1)
    final_objective = hierarchy_mask * values[..., 2]
    scalarized_objectives.append(final_objective)

    expected = torch.stack(scalarized_objectives, dim=-1)
    expected = torch.sum(expected, dim=-1)

    assert torch.allclose(result, expected, atol=1e-4), f"Expected {expected}, but got {result}"
