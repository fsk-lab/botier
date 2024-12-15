from typing import Optional
import torch

from botier.utils import smoothmin, logistic


class HierarchyScalarizer(torch.nn.Module):
    """
    Implementation of a flexible, hierarchy-based scalarizer for scalarizing a set of objective values according to a
    hierarchy of objectives. The scalarizer applies the hierarchy scalarization relative to threshold values :math:`t_i`
    according to

    .. math::
        h(x) = \\sum_{i=1}^{N}{\\min(g_i(x), t_i) \\cdot \\prod_{j=1}^{i-1}{H(g_j(x)-t_j)}} + f_{\\mathrm{fin}}(x) \\cdot \\prod_{i=1}^{N}{H(g_i(x)-t_i)}

    where :math:`H(x)` is the Heaviside function. Both the Heaviside function and the min function are approximated by
    continuous, differentiable variants (see model.py for further details).

    Returns a reduced tensor of scalarized objective values.

    Args:
        final_objective_idx (int, optional): An integer defining which objective in :code:`objectives` should be optimized if the
                             satisfaction criteria are met for all objectives. Defaults to 0 (i.e. the first objective
                             in the hierarchy).
        k (float, optional): The smoothing factor applied to the smoothened, differentiable versions of the min and Heaviside
           functions
    """
    def __init__(
            self,
            final_objective_idx: Optional[int] = 0,
            k: Optional[float] = 1E2
    ):
        super().__init__()
        self._final_obj_idx = final_objective_idx
        self._k = k

    def forward(self, values: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the forward pass.

        Args:
            values (torch.Tensor): A :code:`... x n_obj`-dim tensor of values to be scalarized
            thresholds (torch.Tensor): A :code:`n_obj`-dim tensor of thresholds for each objective.

        Returns:
            torch.Tensor: A :code:`...`-dim Tensor of scalarized objective values.
        """
        hierarchy_masks, scalarized_objectives = [], []

        for idx in range(values.shape[-1]):
            hierarchy_masks.append(logistic(values[..., idx], thresholds[idx], self._k))  # list of `...`-dim tensors
            scalarized_objective = smoothmin(values[..., idx], thresholds[idx], self._k)  # shape: `...`

            # for all but the first objective apply the "hierarchy mask" to set the objective values to 0 in all regions
            # but the ones where the objective is "dominant"
            if idx > 0:
                hierarchy_mask = torch.stack(hierarchy_masks[:idx], dim=-1)  # shape: `... x idx`
                hierarchy_mask = torch.prod(hierarchy_mask, dim=-1)  # shape: `...`
                scalarized_objective = hierarchy_mask * scalarized_objective  # shape: `...`

            scalarized_objectives.append(scalarized_objective)

        # do the same for the final objective (i.e. the one that should be optimized if all thresholds are met)
        hierarchy_mask = torch.stack(hierarchy_masks, dim=-1)  # shape: `... x idx`
        hierarchy_mask = torch.prod(hierarchy_mask, dim=-1)  # shape: `...`
        final_objective = hierarchy_mask * values[..., self._final_obj_idx]  # shape: `...`
        scalarized_objectives.append(final_objective)

        # Sum the scalarized objectives along the `num_objectives` dimension
        scalarized_objectives = torch.stack(scalarized_objectives, dim=-1)  # shape: `... x (n_obj + 1)`
        return torch.sum(scalarized_objectives, dim=-1)  # shape: `...`
