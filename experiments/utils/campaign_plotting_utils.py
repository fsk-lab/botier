import copy
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from botier import AuxiliaryObjective, HierarchyScalarizationObjective


LINESTYLES = [
    (0, ()),
    (0, (5, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1))
]


def calculate_objective_values(
        data: dict[str, Tensor],
        objectives: list[AuxiliaryObjective],
        normalize: bool = True,
        budget: int = None
) -> Tensor:
    """
    Takes the data from a specific run, and calculates the values of each objective.

    Args:
        data: A dictionary containing the data from a run
        objectives: A list of `AuxiliaryObjective` instances
        normalize: True if the objective values should be normalized
        budget: The maximum number of evaluations to consider

    Returns:
        A tensor of objective values (shape `iterations x budget x num_objectives`)
    """
    x, y = data["x"], data["y"]  # shape: `iterations x budget x dim`, `iterations x budget x num_outputs`

    if budget is not None:
        if x.shape[1] > budget:
            x = x[:, :budget, :]
            y = y[:, :budget, :]

    return torch.stack([obj(y, x, normalize=normalize) for obj in objectives], dim=-1)
    # shape: `iterations x budget x num_objectives`


def calculate_satisfaction_likelihood(
        obj_values: Tensor,  # shape: `iterations x budget x num_objectives`
        objectives: list[AuxiliaryObjective],
        normalize: bool = True
) -> Tensor:
    """
    For each time step, calculates the likelihood that all objectives are satisfied. Returns a (`budget`) tensor of
    likelihoods. Assumes that the order in `objectives` corresponds to the order of the objectives in `obj_values` along
    the last dimension.

    Args:
        obj_values: A tensor of objective values (shape `iterations x budget x num_objectives`)
        objectives: A list of `AuxiliaryObjective` instances
        normalize: True if the objective values should be normalized before checking for satisfaction

    Returns:
        A tensor of satisfaction likelihoods (shape `budget`)
    """
    if normalize:
        satisfaction = [obj_values[..., i] >= 1.0 for i, obj in enumerate(objectives)]
    else:
        satisfaction = [obj_values[..., i] >= obj.threshold for i, obj in enumerate(objectives)]

    satisfaction = torch.stack(satisfaction, dim=-1)  # shape: `iterations x budget x len(objectives)`, dtype: `bool`
    overall_satisfaction = torch.all(satisfaction, dim=-1)  # shape: `iterations x budget`, dtype: `bool`
    cumulative_satisfaction = overall_satisfaction.cummax(dim=1).values  # shape: `iterations x budget`, dtype: `bool`

    return torch.sum(cumulative_satisfaction, dim=0) / obj_values.shape[0]  # shape: `budget`, dtype: `float`


def calculate_satisfaction_counts(
        obj_values: Tensor,  # shape: `iterations x budget x num_objectives`
        objectives: list[AuxiliaryObjective],
        normalize: bool = True
) -> Tensor:
    """
    For each iteration and time step, calculates the overall count of data points that satisfy all objectives. Assumes
    that the order in `objectives` corresponds to the order of the objectives in `obj_values` along the last dimension.

    Args:
        obj_values: A tensor of objective values (shape `iterations x budget x num_objectives`)
        objectives: A list of `AuxiliaryObjective` instances
        normalize: True if the objective values should be normalized before checking for satisfaction

    Returns:
        A tensor of satisfaction counts (shape `iterations x budget`)
    """
    if normalize:
        satisfaction = [obj_values[..., i] >= obj.normalized_threshold for i, obj in enumerate(objectives)]
    else:
        satisfaction = [obj_values[..., i] >= obj.threshold for i, obj in enumerate(objectives)]

    satisfaction = torch.stack(satisfaction, dim=-1)  # shape: `iterations x budget x len(objectives)`, dtype: `bool`
    overall_satisfaction = torch.all(satisfaction, dim=-1)  # shape: `iterations x budget`, dtype: `bool`
    return torch.cumsum(overall_satisfaction, dim=1)  # shape: `iterations x budget`, dtype: `int`


def calculate_confidence_region(
        values: Tensor,
        confidence_interval: float = None,
        num_stds: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calculates the mean, upper and lower bounds of the confidence region for the given values, passed as a (`iterations`
    x `budget`) tensor. Returns a tuple of three tensors.

    If the `confidence_interval` is given, the upper and lower bounds are calculated as the `confidence_interval`-th
    percentile of the values. If `confidence_interval` is `None`, the upper and lower bounds are calculated as the mean
    plus and minus `num_stds` standard deviations of the values.

    Args:
        values: A tensor of values (shape `iterations x budget`)
        confidence_interval: The confidence interval to use for the bounds
        num_stds: The number of standard deviations to use for the bounds

    Returns:
        Tensor: The mean of the confidence region (shape `budget`)
        Tensor: The lower bound of the confidence region (shape `budget`)
        Tensor: The upper bound of the confidence region (shape `budget`)
    """
    values = values.double().squeeze(-1)  # shape: `iterations x budget`
    mean = torch.mean(values, dim=0)  # shape: `budget`

    if confidence_interval is None:
        std_err = torch.std(values, dim=0) / torch.sqrt(torch.tensor(values.shape[0]))
        return mean, mean - num_stds * std_err, mean + num_stds * std_err

    else:
        lower = torch.quantile(values, 1-confidence_interval, dim=0)
        upper = torch.quantile(values, confidence_interval, dim=0)
        return mean, lower, upper


def plot_satisfaction_likelihoods(ax, objectives: list[AuxiliaryObjective], *args, budget: int = None, show_labels: bool = True):
    """
    Plots the likelihood of satisfaction for all objectives for each run.

    Args:
        ax: The matplotlib axis to plot on
        objectives: A list of `AuxiliaryObjective` instances
        *args: A list of dictionaries containing the data for each run
        budget: The maximum number of evaluations to consider
        show_labels: True if the axis labels should be shown
    """
    args = copy.deepcopy(args)

    for run in args:
        data = torch.load(run.pop("file"))
        obj_values = calculate_objective_values(data, objectives, normalize=True, budget=budget)

        for i in range(len(objectives)):
            likelihood = calculate_satisfaction_likelihood(obj_values, objectives[:i+1], normalize=True)
            likelihood = torch.cat([torch.tensor([0.0]), likelihood])

            if i > 0:
                run.pop("label", None)

            ax.plot(torch.arange(likelihood.shape[0]).numpy(), likelihood.numpy(), linestyle=LINESTYLES[i], **run)

            ax.set_xlim(None, max(likelihood.shape[0] - 1, ax.get_xlim()[1]))

    if show_labels:
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Satisfaction Likelihood")
        ax.legend(fontsize="small")

    ax.set_ylim(-0.01, 1.02)
    [x.set_linewidth(0.5) for x in ax.spines.values()]
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))


def plot_satisfaction_counts(ax, objectives: list[AuxiliaryObjective], *args, confidence_interval: float = None, num_stds: float = 1.0, budget: int = None, show_labels: bool = True):
    """
    Plots the likelihood of satisfaction for all objectives for each run.

    Args:
        ax: The matplotlib axis to plot on
        objectives: A list of `AuxiliaryObjective` instances
        *args: A list of dictionaries containing the data for each run
        confidence_interval: The confidence interval to use for the bounds
        num_stds: The number of standard deviations to use for the bounds
        budget: The maximum number of evaluations to consider
        show_labels: True if the axis labels should
    """
    args = copy.deepcopy(args)

    for run in args:
        data = torch.load(run.pop("file"))
        obj_values = calculate_objective_values(data, objectives, normalize=True, budget=budget)

        for i in range(len(objectives)):
            counts = calculate_satisfaction_counts(obj_values, objectives[:i+1], normalize=True)
            mean, lower, upper = calculate_confidence_region(counts, confidence_interval, num_stds)

            if i > 0:
                run.pop("label", None)

            ax.plot(torch.arange(mean.shape[0]).numpy(), mean.numpy(), linestyle=LINESTYLES[i], **run)
            ax.fill_between(torch.arange(mean.shape[0]).numpy(), lower.numpy(), upper.numpy(), alpha=0.2, **run)

            ax.set_xlim(0, max(mean.shape[0] - 1, ax.get_xlim()[1]))

    if show_labels:
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Number of Satisfactory Data Points")
        ax.legend(fontsize="small")

    ax.set_ylim(-0.01, None)
    [x.set_linewidth(0.5) for x in ax.spines.values()]


def plot_scalarized_objective(ax, objectives: list[AuxiliaryObjective], *args, accumulate: bool = True, confidence_interval: float = None, num_stds: float = 1.0, budget: int = None, show_labels: bool = True):
    """
    Plots the scalarized objective value for each run.

    Args:
        ax: The matplotlib axis to plot on
        objectives: A list of `AuxiliaryObjective` instances
        *args: A list of dictionaries containing the data for each run
        accumulate: True if the maximum value should be accumulated
        confidence_interval: The confidence interval to use for the bounds
        num_stds: The number of standard deviations to use for the bounds
        budget: The maximum number of evaluations to consider
        show_labels: True if the axis labels should be shown
    """
    args = copy.deepcopy(args)

    for run in args:

        data = torch.load(run.pop("file"))
        scalarizer = HierarchyScalarizationObjective(objectives, normalized_objectives=True)
        scores = scalarizer(data["y"], data["x"])  # shape: `iterations x budget`

        if budget:
            if scores.shape[1] > budget:
                scores = scores[:, :budget]

        if accumulate:
            scores = torch.cummax(scores, dim=1).values  # shape: `iterations x budget`

        mean, lower, upper = calculate_confidence_region(scores, confidence_interval, num_stds)

        ax.plot(torch.arange(mean.shape[0]).numpy(), mean.numpy(), **run)
        run.pop("label", None)
        ax.fill_between(torch.arange(mean.shape[0]).numpy(), lower.numpy(), upper.numpy(), alpha=0.2, **run)

        ax.set_xlim(0, max(mean.shape[0] - 1, ax.get_xlim()[1]))

    if show_labels:
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Scalarized Objective Value")
        ax.legend(fontsize="small")

    ax.set_ylim(-0.01, None)
    [x.set_linewidth(0.5) for x in ax.spines.values()]
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))


def plot_objective_at_optimum(ax, objectives: list[AuxiliaryObjective], obj_idx: int, *args, confidence_interval: float = None, num_stds: float = 1.0, budget: int = None, show_labels: bool = True):
    """
    For each run, identifies the best data point according to the scalarized objective function, and plots the value of
    the specified objective at that point.

    Args:
        ax: The matplotlib axis to plot on
        objectives: A list of `AuxiliaryObjective` instances
        obj_idx: The index of the objective to plot
        *args: A list of dictionaries containing the data for each run
        confidence_interval: The confidence interval to use for the bounds
        num_stds: The number of standard deviations to use for the bounds
        budget: The maximum number of evaluations to consider
        show_labels: True if the axis labels should be shown
    """
    args = copy.deepcopy(args)

    for run in args:

        data = torch.load(run.pop("file"))
        obj_values = calculate_objective_values(data, objectives, normalize=False, budget=budget)  # shape: `iterations x budget x num_objectives`
        scalarizer = HierarchyScalarizationObjective(objectives, normalized_objectives=True)
        scores = scalarizer(data["y"], data["x"])  # shape: `iterations x budget`

        if budget:
            if scores.shape[1] > budget:
                obj_values = obj_values[:, :budget, :]
                scores = scores[:, :budget]

        optimum_indices = torch.cummax(scores, dim=1).indices  # shape: `iterations x budget`
        optimum_values = torch.gather(obj_values[..., obj_idx], 1, optimum_indices)  # shape: `iterations x budget`

        mean, lower, upper = calculate_confidence_region(optimum_values, confidence_interval, num_stds)

        ax.plot(torch.arange(mean.shape[0]).numpy(), mean.numpy(), **run)
        run.pop("label", None)
        ax.fill_between(torch.arange(mean.shape[0]).numpy(), lower.numpy(), upper.numpy(), alpha=0.2, **run)

        ax.set_xlim(0, max(mean.shape[0] - 1, ax.get_xlim()[1]))

    if show_labels:
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel(f"Objective Value")
        ax.legend(fontsize="small")

    [x.set_linewidth(0.5) for x in ax.spines.values()]