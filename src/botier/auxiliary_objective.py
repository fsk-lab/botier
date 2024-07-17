from typing import Tuple, Optional, Callable
import torch
from torch import Tensor


class AuxiliaryObjective(torch.nn.Module):
    """
    Class for auxiliary objectives that can be used in

    Args:
        maximize: True if the objective should be maximized, False if it should be minimized.
        calculation: A callable that computes the value of the auxiliary objective from experiment outputs and inputs.
                     The function should take the following arguments:
                        - Y: A `... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the shape
                          is `sample_shape x batch_shape x q x m`)
                        - X: A `...`-dim tensor of inputs. Relevant only if the objective depends on the inputs
                             explicitly.
        worst_value: The worst possible value of the objective. Required in case the objective should be MinMax-scaled
                     between `worst_value` and `abs_threshold`.
        best_value: The best possible value of the objective. Required in case the objective should be MinMax-scaled
                     between `worst_value` and `abs_threshold`, but only `rel_threshold` is provided.
        abs_threshold: An absolute threshold value for the objective.
        rel_threshold: A threshold value for the objective. Required only if abs_threshold is not provided.
        output_index: The index of the output dimension of the samples that the objective depends on.
    """
    def __init__(
            self,
            maximize: bool = True,
            calculation: Optional[Callable] = None,
            best_value: Optional[float] = None,
            worst_value: Optional[float] = None,
            abs_threshold: Optional[float] = None,
            rel_threshold: Optional[float] = None,
            output_index: Optional[int] = None
    ):
        super().__init__()

        # Sets the function to calculate the objective from x and y.
        if calculation is None:
            if output_index is None:
                raise ValueError("Either a calculation function or an output index must be provided.")
            else:
                self.function = lambda y, x: y[..., output_index]
        else:
            self.function = calculation

        # Set the boundary values
        if best_value is not None and worst_value is not None:
            if maximize is True and best_value < worst_value:
                raise ValueError("For maximization problems, the best possible value must be greater than the worst "
                                 "possible value.")
            elif maximize is False and best_value > worst_value:
                raise ValueError("For minimization problems, the best possible value must be smaller than the worst "
                                 "possible value.")
        self.best_value, self.worst_value = best_value, worst_value

        # if `abs_threshold` is not provided, the threshold is calculated from the relative threshold and the best and
        # worst values
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
            self.normalizable = True

        # if `abs_threshold` is provided, the threshold is set to this value after checking compatibility with
        # the best and worst values
        else:
            if worst_value is not None:
                if maximize is True and abs_threshold < worst_value:
                    raise ValueError("For maximization problems, the satisfaction threshold must be greater than the "
                                     "worst possible value.")
                elif maximize is False and abs_threshold > worst_value:
                    raise ValueError("For minimization problems, the satisfaction threshold must be smaller than the "
                                     "worst possible value.")
                self.normalizable = True
            else:
                self.normalizable = False
            if best_value is not None:
                if maximize is True and abs_threshold > best_value:
                    raise ValueError("For maximization problems, the satisfaction threshold must be smaller than the "
                                     "best possible value.")
                elif maximize is False and abs_threshold < best_value:
                    raise ValueError("For minimization problems, the satisfaction threshold must be greater than the "
                                     "worst possible value.")

            self.abs_threshold = abs_threshold

    def forward(self, Y: Tensor, X: Optional[Tensor], normalize: bool = True) -> Tensor:
        """
        Computes the scalarized auxiliary objective function for the given samples and data points.

        Args:
            Y: A `... x m`-dim Tensor of y values (e.g. from a model posterior, in this case, the
                     shape is `sample_shape x batch_shape x q x m`)
            X: A `...`-dim tensor of inputs. Relevant only if the objective depends on the inputs
               explicitly.
            normalize: True if the objective should be MinMax-scalarized returned to a [0, 1] scale, where 0 corresponds
                       to the worst possible value and 1 to the satisfaction threshold.

        Returns:
            A `...`-dim tensor of auxiliary objective values.
        """
        values = self.function(Y, X)

        # Performs a MinMax normalization
        if normalize and self.normalizable:
            values = (values - self.worst_value) / (self.threshold - self.worst_value)

        return values

    @property
    def bounds(self) -> Tuple[float, float]:
        """
        Provides the bounds of the objective as a Tuple of floats.

        Returns:
            float: Lower bound (i.e. the "worst" possible value)
            float: Upper bound (i.e. the "best" possible value).
        """
        return self.worst_value, self.best_value

    @property
    def threshold(self) -> float:
        """
        Returns the absolute satisfaction threshold value of the objective.
        """
        return self.abs_threshold
