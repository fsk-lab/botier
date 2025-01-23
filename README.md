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

from botier import AuxiliaryObjective HierarchicalScalarizationObjective

# define the 'auxiliary objectives' that eventually make up the overall optimization objective
objectives = [
    AuxiliaryObjective(output_index=0, abs_threshold=0.5, best_value=1.0, worst_value=-1.0)
    AuxiliaryObjective(known=True, calculation=lambda y, x, idx: x[..., 0], abs_theshold=0.0, best_value=0.0, worst_value=1.0)
]
global_objective = HierarchyScalarizationObjective(objectives, k=1E2, normalized_objectives=True)

# generate some training data
train_x = torch.rand(5, 1)
train_y = torch.sin(2 * torch.pi * x)

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
```

For more detailed usage examples, see ```examples```.

## Contributors

Felix Strieth-Kalthoff ([@felix-s-k](https://github.com/felix-s-k)), Mohammad Haddadnia ([@mohaddadnia](https://github.com/Mohaddadnia)) 
