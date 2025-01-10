from typing import List, Optional
import torch
from torch import Tensor

from chimera import Chimera
from botier import AuxiliaryObjective


class ChimeraWrapper(torch.nn.Module):
    """
    A wrapper around the Chimera scalarizer (Häse et al., Chem. Sci., 2018) that allows for easy integration into
    BoTier / BoTorch workflows.

    ATTN: Chimera is not differentiable, so this wrapper can only be used as a black-box scalarizer (but not as a
          composite objective in BoTorch)!
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
        self._k = k

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
            X: A `... x d`-dim tensor of inputs. Just here for consistency with other scalarizers.

        Returns:
            Tensor: A `...`-dim Tensor of scalarized objective values.
        """
        objective_values = self.calculate_objective_values(samples, X, normalize=self._norm)  # shape: `... x num_objectives`

        if self._norm is True:
            thresholds = [1.0 for _ in self.objectives]  # len: `num_objectives`
        else:
            thresholds = [obj.threshold for obj in self.objectives]  # len: `num_objectives`

        chim = Chimera(
            tolerances=thresholds,
            absolutes=[not self._norm for _ in self.objectives],
            goals=["max" if obj.maximize or self._norm else "min" for obj in self.objectives],
            softness=1.0 / self._k
        )

        objective_values = objective_values.cpu().detach().numpy()
        scores = -1.0 * chim.scalarize(objective_values)

        return torch.tensor(scores).to(samples)
