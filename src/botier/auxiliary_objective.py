from typing import Optional, Callable
import torch


class AuxiliaryObjective(torch.nn.Module):
    """
    Class for auxiliary objectives that can be used for creating composite objectives for Multi-Objective Bayesian
    Optimization. The class provides a callable that computes the auxiliary objective from the inputs and outputs of
    an experiment in an auto-differentiable way.

    Returns a modified tensor of objective values that can be used for calculating hierarchical acquisition functions.
    The values of the auxiliary objective are always non-negative, with 0 corresponding to the worst possible value. The
    values can be optionally scaled to the [0, 1]-interval (with 1 being the satisfaction threshold).

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
            self.lower_bound, self.upper_bound = lower_bound, upper_bound
        else:
            self.lower_bound, self.upper_bound = None, None

        # if `abs_threshold` is not provided, the threshold is calculated from the relative threshold and the bounds
        if abs_threshold is None:
            if rel_threshold is None:
                raise ValueError("Either an absolute or a relative satisfaction threshold must be given for an "
                                 "objective.")
            if self.upper_bound is None or self.lower_bound is None:
                raise ValueError("If only a relative threshold (instead of an absolute threshold) is provided, the "
                                 "upper and lower bounds of the objective must be given as well.")
            if rel_threshold < 0.0 or rel_threshold > 1.0:
                raise ValueError("The relative satisfaction threshold must be between 0 and 1.")
            if self.maximize:
                self.abs_threshold = self.lower_bound + rel_threshold * (self.upper_bound - self.lower_bound)
            else:
                self.abs_threshold = self.upper_bound - rel_threshold * (self.upper_bound - self.lower_bound)

        # if `abs_threshold` is provided, the threshold is set after checking compatibility with the bounds
        else:
            if self.normalizable:
                if not (self.lower_bound <= abs_threshold <= self.upper_bound):
                    raise ValueError("The satisfaction threshold must be within the provided lower and upper bounds.")

            self.abs_threshold = abs_threshold

    def forward(self, Y: torch.Tensor, X: Optional[torch.Tensor], normalize: bool = True) -> torch.Tensor:
        """
        Computes the scalarized auxiliary objective function for the given samples and data points. Automatically
        handles the optimization direction so that the resulting values can be treated as an optimization problem.
        Returns strictly positive values which can be used for additive scalarization functions.

        If `normalize` is True, it linearly transforms the data such that the worst possible value corresponds to 0, and
        the satisfaction threshold corresponds to 1. If `normalize` is False, it linearly transforms the data such that
        the worst possible value corresponds to 0, and the original value range is preserved.

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

        if normalize:
            if self.normalizable:
                best, worst = (self.upper_bound, self.lower_bound) if self.maximize else (self.lower_bound, self.upper_bound)
            else:
                upper, lower = values.flatten().max().item(), values.flatten().min().item()
                best, worst = (upper, lower) if self.maximize else (lower, upper)

            # apply MinMax scalarization, and clamp the values out of the provided value range
            values_scaled = (values - worst) / (self.threshold - worst)
            best_scaled = (best - worst) / (self.threshold - worst)
            values_scaled = torch.clamp(values_scaled, 0.0, best_scaled)

        else:
            values_transformed = values if self.maximize else -values

            if self.normalizable:
                if self.maximize:
                    values_scaled = values_transformed - self.lower_bound
                    values_scaled = torch.clamp(values_scaled, 0.0, self.upper_bound - self.lower_bound)
                else:
                    values_scaled = values_transformed + self.upper_bound
                    values_scaled = torch.clamp(values_scaled, 0.0, self.lower_bound + self.upper_bound)
            else:
                lower_transformed = values_transformed.flatten().min().item()
                values_scaled = values_transformed - lower_transformed

        return values_scaled

    @property
    def threshold(self) -> float:
        """
        Returns the absolute satisfaction threshold value of the objective.
        """
        return self.abs_threshold

    @property
    def normalizable(self) -> bool:
        """
        Returns True if the objective can be MinMax-normalized, i.e. if both the upper and lower bounds are provided.
        """
        return self.lower_bound is not None and self.upper_bound is not None
