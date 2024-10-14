from typing import Optional, List
import torch
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionObjective

from botier import AuxiliaryObjective
from botier.utils import smoothmin


class PenaltyScalarizer(torch.nn.Module):
    """
    Implementation of the penalty-based hierarchical scalarizer for scalarizing a set of objectives, categorized into a
    primary objective and multiple secondary objectives. Follows the general implementation described in Walker et al.
    (React. Chem. Eng. 2017, 2, 785).

    Args:
        seed_optimum: The optimum value of the primary objective obtained from the seed data
        seed_median_deviation: The median of the absolute deviations of the primary objective from the optimum value,
                               as obtained from the seed data
        k: [optional] The smoothing factor applied to the smoothened, differentiable versions of the min and Heaviside
           functions
    """
    def __init__(
            self,
            k: Optional[float] = 1E2
    ):
        super().__init__()
        self._k = k

    def forward(self, values: Tensor, thresholds: Tensor) -> Tensor:
        """
        Implementation of the forward pass.

        Args:
            values: A `... x n_obj`-dim tensor of values to be scalarized
            thresholds: A `n_obj`-dim tensor of thresholds for each objective.

        Returns:
            Tensor: A `...`-dim Tensor of scalarized objective values.
        """
        penalties = []
        for idx in range(1, values.shape[-1]):
            penalties.append(smoothmin(values[..., idx], thresholds[idx], self._k).pow(2))  # shape: `...`
        penalty_sum = torch.sum(torch.stack(penalties, dim=-1), dim=-1)  # shape: `...`

        return values[..., 0] / (0.1 + values[..., 0].abs()) + penalty_sum


class PenaltyScalarizationObjective(MCAcquisitionObjective):
    """
    Implementation of the PenaltyScalarizer as a MCAcquisitionObjective for botorch's MonteCarlo acquisition function
    framework.

    Takes a `sample_shape x batch_shape x q x m` tensor (as returned by botorch's posterior sampling routine), where...
        - `sample_shape` is the number (or shape) of posterior samples
        - `batch_shape` is the number of x locations to be evaluated simultaneously (e.g. during acquisition function
          optimization)
        - `q` is the number of x locations for which a joint acquisition function optimum should be obtained (batch
          acquisition)
        - `m` is the number of model outputs

    First calculates the objective values g_i(x) for each of the N objective from the inputs and model predictions (as
    specified in AuxiliaryObjective.forward(...)). Second, applies the hierarchy scalarization relative to threshold
    values t_i (see PenaltyScalarizer for further details).

    Returns a reduced `sample_shape x batch_shape x q` tensor of scalarized objective values.

    Args:
        objectives: A list of AuxiliaryObjective objects, defining the value ranges and the satisfaction threshold for
                    each objective.
        final_objective_idx: [optional] An integer defining which objective in `objectives` should be optimized if the
                             satisfaction criteria are met for all objectives. Defaults to 0 (i.e. the first objective
                             in the hierarchy.
        normalized_objectives: True if the objectives should each be normalized on a [0, 1] scale before applying the
                               hierarchy scalarization
        k: [optional] The smoothing factor applied to the smoothened, differentiable versions of the min and Heavyside
           functions
    """
    def __init__(
            self,
            objectives: List[AuxiliaryObjective],
            normalized_objectives: bool = True,
            k: Optional[float] = 1E2
    ):
        super().__init__()
        self.scalarizer = PenaltyScalarizer(k)
        self.objectives = objectives
        self._norm = normalized_objectives

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
            thresholds = torch.tensor([1.0 for obj in self.objectives], device=samples.device)  # shape: `num_objectives`
        else:
            thresholds = torch.tensor([obj.threshold for obj in self.objectives], device=samples.device)  # shape: `num_objectives`

        return self.scalarizer(objective_values, thresholds)
