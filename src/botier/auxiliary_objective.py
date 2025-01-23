from typing import Tuple, Optional, Callable
import torch


class AuxiliaryObjective(torch.nn.Module):
    """
    Class for auxiliary objectives that can be used for creating composite objectives for Multi-Objective Bayesian
    Optimization. The class provides a callable that computes the auxiliary objective from the inputs and outputs of
    an experiment in an auto-differentiable way.

    Provides automatic handling of MinMax-scaling of the objective values to a [0, 1] range, which is very useful when
    defining differentiable composite_objectives.

    Args:
        maximize (bool): True if the objective should be maximized, False if it should be minimized.
        calculation (Callable, optional): A callable that computes the value of the auxiliary objective from experiment outputs and inputs.
                     The function should take the following arguments:
                        - Y: A :code:`... x m`-dim Tensors of samples (e.g. from a model posterior, in this case, the shape
                          is :code:`sample_shape x batch_shape x q x m`)
                        - X: A :code:`...`-dim tensor of inputs. Relevant only if the objective depends on the inputs explicitly.
        lower_bound (float, optional): The lower bound value of the objective. Corresponds to the worst value for maximization problems
                     and the best value for minimization problems. Required in case the objective should be MinMax-
                     scaled between :code:`worst_value` and :code:`abs_threshold`. All values outside the window :code:`[lower_bound,
                     upper_bound]` are clipped to the respective bound.
        upper_bound (float, optional): The upper bound value of the objective. Corresponds to the best value for maximization problems and
                     the worst value for minimization problems. Required in case the objective should be MinMax-scaled
                     between :code:`worst_value` and :code:`abs_threshold`, but only :code:`rel_threshold` is provided. During MinMax-
                     scaling, all values outside the window :code:`[lower_bound, upper_bound]` are clipped to the respective
                     bound.
        abs_threshold (float, optional): An absolute threshold value for the objective.
        rel_threshold (float, optional): A threshold value for the objective. Required only if :code:`abs_threshold` is not provided.
        output_index (int, optional): The index of the output dimension of the samples that the objective depends on.
    """
    def __init__(
            self,
            maximize: bool = True,
            calculation: Optional[Callable] = None,
            upper_bound: Optional[float] = None,
            lower_bound: Optional[float] = None,
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
        self.maximize = maximize
        if upper_bound is not None and lower_bound is not None:
            if upper_bound < lower_bound:
                raise ValueError("The upper bound must be greater than the lower bound.")
            if maximize is True:
                self.best_value, self.worst_value = upper_bound, lower_bound
            else:
                self.best_value, self.worst_value = lower_bound, upper_bound
        else:
            self.best_value, self.worst_value = None, None

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
            self.abs_threshold = self.worst_value + rel_threshold * (self.best_value - self.worst_value)
            self.normalizable = True

        # if `abs_threshold` is provided, the threshold is set to this value after checking compatibility with
        # the best and worst values
        else:
            if self.best_value is not None:
                if maximize is True and abs_threshold < self.worst_value:
                    raise ValueError("For maximization problems, the satisfaction threshold must be greater than the "
                                     "worst possible value.")
                elif maximize is False and abs_threshold > self.worst_value:
                    raise ValueError("For minimization problems, the satisfaction threshold must be smaller than the "
                                     "worst possible value.")
                self.normalizable = True
            else:
                self.normalizable = False
            if self.best_value is not None:
                if maximize is True and abs_threshold > self.best_value:
                    raise ValueError("For maximization problems, the satisfaction threshold must be smaller than the "
                                     "best possible value.")
                elif maximize is False and abs_threshold < self.best_value:
                    raise ValueError("For minimization problems, the satisfaction threshold must be greater than the "
                                     "best possible value.")

            self.abs_threshold = abs_threshold

    def forward(self, Y: torch.Tensor, X: Optional[torch.Tensor], normalize: bool = True) -> torch.Tensor:
        """
        Computes the scalarized auxiliary objective function for the given samples and data points.

        Args:
            Y (torch.Tensor): A :code:`... x m`-dim Tensor of y values (e.g. from a model posterior, in this case, the
                     shape is :code:`sample_shape x batch_shape x q x m`)
            X (torch.Tensor, optional): A :code:`...`-dim tensor of inputs. Relevant only if the objective depends on the inputs
               explicitly.
            normalize (bool): True if the objective should be MinMax-scalarized returned to a [0, 1] scale, where 0 corresponds
                       to the worst possible value and 1 to the satisfaction threshold.

        Returns:
            A :code:`...`-dim tensor of auxiliary objective values.
        """
        values = self.function(Y, X)

        # Performs a MinMax normalization
        if normalize and self.normalizable:
            values = (values - self.worst_value) / (self.threshold - self.worst_value)

            if self.best_value is None:
                values = torch.clamp(values, 0.0)
            else:
                best_value = (self.best_value - self.worst_value) / (self.threshold - self.worst_value)
                values = torch.clamp(values, 0.0, best_value)

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
