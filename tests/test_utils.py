import torch
from botier.utils import smoothmin, logistic


def test_smoothmin():
    x = torch.randn(10)
    x_0 = torch.randn(1)
    k = 1E2

    expected = torch.empty(x.shape)
    for idx, value in enumerate(x):
        min_val = torch.min(x_0, value)
        expected[idx] = min_val

    # x_0 as Tensor
    result = smoothmin(x, x_0, k)
    assert torch.allclose(result, expected, rtol=1e-1), f"Expected {expected}, but got {result}"

    # x_0 as float
    result = smoothmin(x, x_0.item(), k)
    assert torch.allclose(result, expected, rtol=1e-1), f"Expected {expected}, but got {result}"


def test_logistic():
    x = torch.randn(10)
    x_0 = torch.randn(1)
    k = 1E2

    expected = 1 / (1 + torch.exp(-k * (x - x_0)))

    # x_0 as Tensor
    result = logistic(x, x_0, k)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    # x_0 as float
    result = logistic(x, x_0.item(), k)
    assert torch.allclose(result, expected, rtol=1e-2), f"Expected {expected}, but got {result}"
