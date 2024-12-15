from typing import Optional, List

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.acquisition.multi_objective.monte_carlo import MCMultiOutputObjective

from botier.auxiliary_objective import AuxiliaryObjective
from botier.hierarchy_scalarizer import HierarchyScalarizer


class HierarchyScalarizationObjective(MCAcquisitionObjective):
    """
    Implementation of the :class:`HierarchyScalarizer` as a MCAcquisitionObjective for BoTorch's MonteCarlo acquisition function
    framework.
        1) Computes the objective values for each of the N objectives from the inputs and model predictions (as
           specified by the respective :class:`AuxiliaryObjective` objects).
        2) Applies the hierarchy scalarization relative to threshold values :math:`t_i`:

            .. math::
                h(x) = \\sum_{i=1}^{N}{\\min(g_i(x), t_i) \\cdot \\prod_{j=1}^{i-1}{H(g_j(x)-t_j)}}
                        + f_{\\mathrm{fin}}(x) * \\prod_{i=1}^{N}{H(g_i(x)-t_i)}

            (see :class:`HierarchyScalarizer` for further details).

    Takes a :code:`[... x m]` tensor (e.g. a set of posterior samples with :code:`sample_shape x batch_shape x q x m`  returned by
    BoTorch's posterior sampling routine), where :code:`m` is the number of model outputs. Returns a reduced :code:`...` tensor of
    scalarized objective values.

    Args:
        objectives (List[AuxiliaryObjective]): A list of :class:`AuxiliaryObjective` objects, defining the value ranges and the satisfaction threshold for
                    each objective.
        final_objective_idx (int, optional): An integer defining which objective in :code:`objectives` should be optimized if the
                             satisfaction criteria are met for all objectives. Defaults to 0 (i.e. the first objective
                             in the hierarchy).
        normalized_objectives (bool): True if the objectives should each be normalized on a [0, 1] scale (0: worst possible
                               value, 1: threshold) before applying the hierarchy scalarization
        k (float, optional): The smoothing factor applied to the smoothened, differentiable versions of the min and Heavyside
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

    def calculate_objective_values(self, Y: torch.Tensor, X: Optional[torch.Tensor] = None, normalize: bool = True) -> torch.Tensor:
        """
        Calculates the values of each objective from the experiment outputs and inputs.

        Args:
            Y (torch.Tensor): A :code:`... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the shape is
               :code:`sample_shape x batch_shape x q x m`)
            X (torch.Tensor, optional): A :code:`...`-dim tensor of inputs. Relevant only if the objective depends on the inputs explicitly.
            normalize (bool): True if the objective should be MinMax-scalarized returned to a [0, 1] scale, where 0 corresponds
                       to the worst possible value and 1 to the satisfaction threshold.

        Returns:
            Tensor: A :code:`... x n_obj`-dim tensor of objective values.
        """
        if X is not None:
            if Y.dim() != X.dim():
                X = X.expand(*Y.size()[:-1], X.size(dim=-1))

        return torch.stack(
            [obj(Y, X, normalize=normalize) for obj in self.objectives],
            dim=-1
        )  # shape: `... x num_objectives`

    def forward(self, samples: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of the forward pass.

        Args:
            samples (torch.Tensor): A :code:`... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the tensor shape is
                     :code:`sample_shape x batch_shape x q x m`)
            X (torch.Tensor, optional): A :code:`... x d`-dim tensor of inputs. Relevant only if the objective depends on the inputs explicitly.

        Returns:
            Tensor: A :code:`...`-dim Tensor of scalarized objective values.
        """
        objective_values = self.calculate_objective_values(samples, X, normalize=self._norm)  # shape: `... x num_objectives`

        if self._norm is True:
            thresholds = torch.tensor([1.0 for _ in self.objectives]).to(samples)  # shape: `num_objectives`
        else:
            thresholds = torch.tensor([obj.threshold for obj in self.objectives]).to(samples)  # shape: `num_objectives`

        return self.scalarizer(objective_values, thresholds)


class ObjectiveCalculator(MCMultiOutputObjective, HierarchyScalarizationObjective):
    """
    Implementation of a naive objective calculator that can be used as a :class:`MCMultiOutputObjective` for botorch-type
    optimizations, and has access to the :func:`calculate_objective_values` method from :class:`HierarchyScalarizationObjective`.
    """
    def __init__(
            self,
            objectives: List[AuxiliaryObjective],
            final_objective_idx: Optional[int] = 0,
            normalized_objectives: bool = True,
            k: Optional[float] = 1E2,
    ):
        # HierarchyScalarizationObjective.__init__(self, objectives, final_objective_idx, normalized_objectives, k)
        # MCMultiOutputObjective.__init__(self)
        super().__init__(objectives, final_objective_idx=final_objective_idx, normalized_objectives=normalized_objectives, k=k)

    def forward(self, samples: torch.Tensor, X: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.calculate_objective_values(samples, X, normalize=self._norm)
