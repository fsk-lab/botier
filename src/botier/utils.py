from __future__ import annotations
from typing import Union
import torch
from torch import Tensor


def tensor_argument(func):
    """
    Decorator for class / instance methods that ensures that the first argument is a tensor.
    """
    def wrapper(self, x: Tensor, *args, **kwargs):
        if not isinstance(x, Tensor):
            try:
                x = torch.tensor(x)
            except Exception:
                raise ValueError(f"The input argument of {func.__name__} is not a tensor and cannot be converted.")
        return func(self, x, *args, **kwargs)
    return wrapper


def smoothmin(x: Tensor, x_0: Union[float, Tensor], k: float = 1E2) -> Tensor:
    """
    Implementation of a smoothened and differentiable version of the minimum function. For all values in the input
    Tensor, computes the smoothened minimum between the input value and a constant value x_0. The smoothening is
    controlled by the parameter k.

    Mathematically, the function is defined as:
            smoothmin(x, x_0, k) = (x * exp(-k * x) + x_0 * exp(-k * x_0)) / (exp(-k * x) + exp(-k * x_0))

    Args:
        x (torch.Tensor): The input tensor.
        x_0 (float): The constant value to compare the input tensor to.
        k (float): The smoothening parameter (default: 1E2).

    Returns:
        torch.Tensor: The smoothened minimum between the input tensor and the constant value x_0 (same shape as x).
    """
    if not isinstance(x_0, Tensor):
        x_0 = torch.tensor(x_0)
    if not isinstance(k, Tensor):
        k = torch.tensor(k)
    return (x * torch.exp(-k * x) + x_0 * torch.exp(-k * x_0)) / (torch.exp(-k * x) + torch.exp(-k * x_0))


def logistic(x: Tensor, x_0: Union[float, Tensor], k: float = 1E2) -> Tensor:
    """
    Implementation of the logistic function. For all values in the input Tensor, computes the logistic function relative
    to a fixed location x_0.

    Mathematically, the function is defined as:
            logistic(x, x_0, k) = 1 / (1 + exp(-k * (x - x_0)))

    Args:
        x (torch.Tensor): The input tensor.
        x_0 (float): The location of the logistic function.
        k (float): The steepness parameter (default: 1E2).

    Returns:
        torch.Tensor: The logistic function relative to the location x_0 (same shape as x).
    """
    if not isinstance(x_0, Tensor):
        x_0 = torch.tensor(x_0)
    if not isinstance(k, Tensor):
        k = torch.tensor(k)
    return 1 / (1 + torch.exp(-k * (x - x_0)))
