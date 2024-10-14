r"""
The `botier` package provides a flexible framework to express
hierarchical user preferences over both experiment inputs and outputs.

`botier` is a lightweight plug-in for `botorch`, and can be readily
integrated with the botorch ecosystem for Bayesian Optimization.
"""

from .auxiliary_objective import AuxiliaryObjective
from .hierarchy_scalarization_objective import HierarchyScalarizationObjective, ObjectiveCalculator
from .hierarchy_scalarizer import HierarchyScalarizer


__all__ = [
    "AuxiliaryObjective",
    "HierarchyScalarizationObjective",
    "HierarchyScalarizer",
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
