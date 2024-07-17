from typing import Optional, List
import torch
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.acquisition.multi_objective.monte_carlo import MCMultiOutputObjective

from .auxiliary_objective import AuxiliaryObjective
from .hierarchy_scalarizer import HierarchyScalarizer


class HierarchyScalarizationObjective(MCAcquisitionObjective):
    """
    Implementation of the HierarchyScalarizer as a MCAcquisitionObjective for botorch's MonteCarlo acquisition function
    framework.
        1) Computes the objective values for each of the N objectives from the inputs and model predictions (as
           specified by the respective AuxiliaryObjective objects).
        2) Applies the hierarchy scalarization relative to threshold values t_i:

           h(x) = \sum_{i=1}^{N}{min(g_i(x), t_i) \cdot \prod_{j=1}^{i-1}{H(g_j(x)-t_j)}}
                  + f_fin(x) * \prod_{i=1}{N}{H(g_i(x)-t_i)}

           (see HierarchyScalarizer for further details).

    Takes a `... x m` tensor (e.g. a set of posterior samples with `sample_shape x batch_shape x q x m`  returned by
    botorch's posterior sampling routine), where `m` is the number of model outputs. Returns a reduced `...` tensor of
    scalarized objective values.

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
        self.objectives = objectives
        self._norm = normalized_objectives
        self.scalarizer = HierarchyScalarizer(final_objective_idx, k)

    def calculate_objective_values(self, Y: Tensor, X: Optional[Tensor] = None, normalize: bool = True) -> Tensor:
        """
        Calculates the values of each objective from the experiment outputs and inputs.

        Args:
            Y: A `... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the shape is
               `sample_shape x batch_shape x q x m`)
            X: A `...`-dim tensor of inputs. Relevant only if the objective depends on the inputs explicitly.
            normalize: True if the objective should be MinMax-scalarized returned to a [0, 1] scale, where 0 corresponds
                       to the worst possible value and 1 to the satisfaction threshold.

        Returns:
            Tensor: A `... x n_obj`-dim tensor of objective values.
        """
        if X is not None:
            if Y.dim() != X.dim():
                X = X.expand(*Y.size()[:-1], X.size(dim=-1))

        return torch.stack(
            [obj(Y, X, normalize=normalize) for obj in self.objectives],
            dim=-1
        )  # shape: `... x num_objectives`

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
        objective_values = self.calculate_objective_values(samples, X, normalize=self._norm)  # shape: `... x num_objectives`

        if self._norm is True:
            thresholds = torch.tensor([1.0 for _ in self.objectives]).to(samples)  # shape: `num_objectives`
        else:
            thresholds = torch.tensor([obj.threshold for obj in self.objectives]).to(samples)  # shape: `num_objectives`

        return self.scalarizer(objective_values, thresholds)
