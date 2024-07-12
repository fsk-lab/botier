from typing import Tuple, Optional, Callable
import torch
from torch import Tensor


class AuxiliaryObjective(torch.nn.Module):
    """
    Class for auxiliary objectives that can be used in

    Args:
        calculation: A callable that computes the auxiliary objective function. The function should take the following
                     arguments:
                        - samples: A `... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the
                          shape is `sample_shape x batch_shape x q x m`)
                        - X: A `...`-dim tensor of inputs. Relevant only if the objective depends on the inputs
                             explicitly.
                        - output_index: The index of the output dimension of the samples that the objective depends on.
        best_value: The best possible value of the objective. Required in case of MinMax normalization.
        worst_value: The worst possible value of the objective. Required in case of MinMax normalization.
        abs_threshold: An absolute threshold value for the objective.
        rel_threshold: A threshold value for the objective. Required only if abs_threshold is not provided.
        known: A boolean flag indicating whether the objective is known or not (i.e. if it is explicitly learned by the
               model).
        output_index: The index of the output dimension of the samples that the objective depends on. Can be None upon
                      instantiation, and can be set later.
    """
    def __init__(
            self,
            calculation: Optional[Callable] = None,
            best_value: Optional[float] = None,
            worst_value: Optional[float] = None,
            abs_threshold: Optional[float] = None,
            rel_threshold: Optional[float] = None,
            known: bool = False,
            output_index: Optional[int] = None
    ):
        super().__init__()

        self.function = calculation if calculation is not None else lambda samples, _, idx: samples[..., idx]

        self.best_value, self.worst_value = best_value, worst_value

        if abs_threshold is None:
            if rel_threshold is None:
                raise ValueError("Either an absolute or a relative satisfaction threshold must be given for an "
                                 "objective.")
            if self.best_value is None or self.worst_value is None:
                raise ValueError("If only a relative threshold (instead of an absolute threshold) is provided, the best"
                                 " and worst values need to be given, too")
            if rel_threshold < 0.0 or rel_threshold > 1.0:
                raise ValueError("The relative satisfaction threshold must be between 0 and 1.")
            self.abs_threshold = worst_value + rel_threshold * (best_value - worst_value)
        else:
            if best_value is not None and worst_value is not None:
                if not (best_value > abs_threshold > worst_value or best_value < abs_threshold < worst_value):
                    raise ValueError("The satisfaction threshold must be within the bounds of `best_value` and "
                                     "`worst_value`.")
            self.abs_threshold = abs_threshold

        self.known = known
        self.output_index = output_index

    def forward(self, samples: Tensor, X: Optional[Tensor], normalize: bool = True) -> Tensor:
        """
        Computes the scalarized auxiliary objective function for the given samples and data points.

        Args:
            samples: A `... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the
                     shape is `sample_shape x batch_shape x q x m`)
            X: A `...`-dim tensor of inputs. Relevant only if the objective depends on the inputs
               explicitly.
            normalize: True if the objective should be returned on a normalized [0, 1] scale

        Returns:
            A `...`-dim tensor of auxiliary objective values.
        """
        if self.known is False and self.output_index is None:
            raise ValueError("The output index must be set for non-a-priori-known auxiliary objectives.")

        values = self.function(samples, X, self.output_index)

        # Performs a simple MinMax normalization
        if normalize is True:
            values = (values - self.worst_value) / (self.best_value - self.worst_value)
            values = values.clamp(min=0, max=1)

        return values

    @property
    def bounds(self) -> Tuple[float, float]:
        """
        Provides the bounds of the objective as a Tuple of floats.

        Returns:
            float: Lower bound (i.e. the "worst" possible value)
            float: Upper bound (i.e. the "best" possible value).
        """
        return self.best_value, self.worst_value

    @property
    def threshold(self) -> float:
        """
        Returns the absolute satisfaction threshold value of the objective.
        """
        return self.abs_threshold

    @property
    def normalized_threshold(self) -> float:
        """
        Returns the MinMax-normalized threshold value of the objective, i.e. the "relative" satisfaction threshold
        between 0 ("worst" objective value) and 1 ("best" objective value)
        """
        return (self.abs_threshold - self.worst_value) / (self.best_value - self.worst_value)
