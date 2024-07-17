from typing import Optional, List
import torch
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionObjective

from .auxiliary_objective import AuxiliaryObjective
from .hierarchy_scalarizer import HierarchyScalarizer


class HierarchyScalarizationObjective(MCAcquisitionObjective):
    """
    Implementation of the HierarchyScalarizer as a MCAcquisitionObjective for botorch's MonteCarlo acquisition function
    framework.

    Takes a `... x m` tensor (e.g. a set of posterior samples with `sample_shape x batch_shape x q x m`  returned by
    botorch's posterior sampling routine), where `m` is the number of model outputs

    First calculates the objective values g_i(x) for each of the N objective from the inputs and model predictions (as
    specified in AuxiliaryObjective.forward(...)). Second, applies the hierarchy scalarization relative to threshold
    values t_i (see HierarchyScalarizer for further details).

    h(x) = \sum_{i=1}^{N}{min(g_i(x), t_i) \cdot \prod_{j=1}^{i-1}{H(g_j(x)-t_j)}} + f_fin(x) * \prod_{i=1}{N}{H(g_i(x)-t_i)}

    Returns a reduced `...` tensor of scalarized objective values.

    Args:
        objectives: A list of AuxiliaryObjective objects, defining the value ranges and the satisfaction threshold for
                    each objective.
        final_objective_idx: [optional] An integer defining which objective in `objectives` should be optimized if the
                             satisfaction criteria are met for all objectives. Defaults to 0 (i.e. the first objective
                             in the hierarchy).
        normalized_objectives: True if the objectives should each be normalized on a [0, 1] scale (0: worst possible
                               value, 1: threshold) before applying the hierarchy scalarization
        k: [optional] The smoothing factor applied to the smoothened, differentiable versions of the min and Heavyside
           functions
    """
    def __init__(
            self,
            objectives: List[AuxiliaryObjective],
            final_objective_idx: Optional[int] = 0,
            normalized_objectives: bool = True,
            k: Optional[float] = 1E2
    ):
        super().__init__()
        self.scalarizer = HierarchyScalarizer(final_objective_idx, k)
        self.objectives = objectives
        self._norm = normalized_objectives

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """
        Implementation of the forward pass.

        Args:
            samples: A `... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the tensor shape is
                     `sample_shape x batch_shape x q x m`)
            X: A `... x d`-dim tensor of inputs. Relevant only if the objective depends on the inputs explicitly.

        Returns:
            Tensor: A `...`-dim Tensor of scalarized objective values.
        """
        if X is not None:
            if samples.dim() != X.dim():
                X = X.expand(*samples.size()[:-1], X.size(dim=-1))

        objective_values = torch.stack(
            [obj(samples, X, normalize=self._norm) for obj in self.objectives],
            dim=-1
        )  # shape: `... x num_objectives`

        if self._norm is True:
            thresholds = torch.tensor([1.0 for _ in self.objectives]).to(samples)  # shape: `num_objectives`
        else:
            thresholds = torch.tensor([obj.threshold for obj in self.objectives]).to(samples)  # shape: `num_objectives`

        return self.scalarizer(objective_values, thresholds)
