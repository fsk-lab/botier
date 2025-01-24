[![workflow](https://github.com/fsk-lab/botier/actions/workflows/ci.yml/badge.svg)](https://github.com/fsk-lab/botier/actions/workflows/ci.yml/badge.svg)
[![coverage](https://img.shields.io/codecov/c/github/fsk-lab/botier)](https://img.shields.io/codecov/c/github/fsk-lab/botier)

[![PyPI - Version](https://img.shields.io/pypi/v/botier?label=PyPI)](https://pypi.org/project/botier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)

# BOTier: Multi-Objective Bayesian Optimization with Tiered Preferences over Experiment Outcomes and Inputs

Next to the "primary" optimization objectives, optimization problems often contain a series of subordinate objectives, which can be expressed as preferences over either the outputs of an experiment, or the experiment inputs (e.g. to minimize the experimental cost). **BoTier** provides a flexible framework to express hierarchical user preferences over both experiment inputs and outputs. The details are described in the corresponding paper. 

```botier```is a lightweight plug-in for ```botorch```, and can be readily integrated with the ```botorch``` ecosystem for Bayesian Optimization. 


## Installation

```botier``` can be readily installed from the Python Package Index (PyPI).

```shell
pip install botier
```

## Usage

The following code snippet shows a minimal example of using the hierarchical scalarization objective 

In this example, our primary goal is to maximize the $\sin(2\pi x)$ function to a value of min. 0.5. If this is satisfied, the value of x should be minimized. 

```python
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

import numpy as np
from matplotlib import pyplot as plt

from botier import AuxiliaryObjective, HierarchyScalarizationObjective

# define the 'auxiliary objectives' that eventually make up the overall optimization objective
objectives = [
    AuxiliaryObjective(output_index=0, abs_threshold=0.5, upper_bound=1.0, lower_bound=-1.0),
    AuxiliaryObjective(maximize=False, calculation=lambda y, x: x[..., 0], abs_threshold=0.0, lower_bound=0.0, upper_bound=1.0),
]
global_objective = HierarchyScalarizationObjective(objectives, k=1E2, normalized_objectives=True)

# generate some training data
train_x = torch.rand(5, 1).double()
train_y = torch.sin(2 * torch.pi * train_x)

budget = 20
for n in range(budget):
    
    # fit a simple BoTorch surrogate model
    surrogate = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
    fit_gpytorch_mll(mll)

    # instantiate a BoTorch Monte-Carlo acquisition function using the botier.HierarchyScalarizationObjective as the 'objective' argument
    acqf = qExpectedImprovement(
        model=surrogate,
        objective=global_objective,
        best_f=torch.max(train_y)
    )

    new_candidate, _ = optimize_acqf(acqf, bounds=torch.tensor([[0.0], [1.0]]), q=1, num_restarts=5, raw_samples=512)


    # evaluate the global objective
    new_candidate_y = torch.sin(2 * torch.pi * new_candidate)

    # update the training points
    train_x = torch.cat([train_x, new_candidate])
    train_y = torch.cat([train_y, new_candidate_y])

    print(f"iteration {n + 1}: candidate={new_candidate.item()}, objective={new_candidate_y.item()}")


plt.plot(np.linspace(0, 1, 100), torch.sin(2 * torch.pi * torch.linspace(0, 1, 100)), label="true function", zorder=0)
plt.scatter(train_x.numpy(), train_y.numpy(), s=25, marker="x", cmap="spring", c=np.arange(len(train_x)), label="selected points")
plt.colorbar()
plt.legend()
plt.show()
```

For more detailed usage examples, see ```examples```.

## Contributors

Felix Strieth-Kalthoff ([@felix-s-k](https://github.com/felix-s-k)), Mohammad Haddadnia ([@mohaddadnia](https://github.com/Mohaddadnia)) 
