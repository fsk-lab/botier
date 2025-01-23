.. _usage-tutorial:

===============
BoTier Tutorial
===============

The following code snippet shows a minimal example of using BoTier's hierarchical scalarization as a composite objective.

In this example, our primary goal is to maximize the :math:`\sin(2 \pi x)` function to a value of min. 0.5. If this is satisfied, the value of x should be minimized.

.. code-block:: python

    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.optim import optimize_acqf

    from botier import AuxiliaryObjective HierarchicalScalarizationObjective

We first define the 'auxiliary objectives' that eventually make up the overall optimization objective:

.. code-block:: python

    objectives = [
        AuxiliaryObjective(output_index=0, abs_threshold=0.5, best_value=1.0, worst_value=-1.0)
        AuxiliaryObjective(known=True, calculation=lambda y, x, idx: x[..., 0], abs_theshold=0.0, best_value=0.0, worst_value=1.0)
    ]
    global_objective = HierarchyScalarizationObjective(objectives, k=1E2, normalized_objectives=True)

The first objective is the first model output (index 0) with an absolute threshold of 0.5. The best value is 1.0, and the worst value is -1.0. The second objective is a known value (known=True) that is calculated as the first input parameter (x) with an absolute threshold of 0.0. The best obtainable value is 0.0, and the worst value is 1.0.
The second objective is only dependent on the input parameters (`known = True`), and is calculated from the input parameter `x` using the function passed as the `calculation` argument.
For a detailed explanation of the `AuxiliaryObjective` class, refer to the `API documentation <../api_reference/botier.auxiliary_objective>`_.

That is it! Now we can generate some training data

.. code-block:: python

    train_x = torch.rand(5, 1)
    train_y = torch.sin(2 * torch.pi * x)

And finally, we can run our optimization campaign using BoTorch

.. code-block:: python

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
