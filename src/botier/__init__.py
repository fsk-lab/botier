r"""
The `BoTier` package provides a flexible framework to express
hierarchical user preferences over both experiment inputs and outputs.

`BoTier` is a lightweight plug-in for `botorch`, and can be readily
integrated with the botorch ecosystem for Bayesian Optimization.
"""

import importlib.metadata as _metadata

from botier.auxiliary_objective import AuxiliaryObjective
from botier.hierarchy_scalarization_objective import HierarchyScalarizationObjective
from botier.hierarchy_scalarizer import HierarchyScalarizer

__version__ = _metadata.version("botier")

__all__ = [
    "AuxiliaryObjective",
    "HierarchyScalarizationObjective",
    "HierarchyScalarizer",
    "__version__",
]


def __dir__():
    return __all__


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(
            f"Module 'botier' has no attribute '{name}'"
        )
