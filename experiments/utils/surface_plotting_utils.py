import itertools
import numpy as np
import torch
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem

from botier import ObjectiveCalculator, AuxiliaryObjective


def plot_function_with_objectives(
        function: MultiObjectiveTestProblem,
        objectives: list[AuxiliaryObjective],
        file_name: str,
        num_points: int = 1E7
):
    """
    Plots the function values and the satisfaction thresholds for the given objectives.

    Args:
        function: The multi-objective test function to plot.
        objectives: A list of auxiliary objectives to evaluate the function on.
        file_name: The name of the file to save the plot to. The file name will be appended with the satisfaction
                     fraction (temporarily) achieved by the objectives.
        num_points: The number of points to sample from the input space
    """
    # draw samples over the entire input space
    sampler = SobolEngine(dimension=function.dim, scramble=True, seed=42)
    grid = sampler.draw(num_points) * (function.bounds[1] - function.bounds[0]) + function.bounds[0]

    # evaluate the function on the grid
    values = function(grid)  # shape: `num_points x r`
    obj_values = ObjectiveCalculator(objectives, normalized_objectives=False)(values, grid)
    obj_values_normal = ObjectiveCalculator(objectives, normalized_objectives=True)(values, grid)

    # calculate the satisfaction of each objective, and the overall satisfaction
    satisfaction = torch.stack([obj_values_normal[..., i] >= 1.0 for i, obj in enumerate(objectives)], dim=-1)
    overall_satisfaction = torch.all(satisfaction, dim=-1)
    satisfactory_points = obj_values[overall_satisfaction, ...]
    satisfaction_fraction = torch.sum(overall_satisfaction).item() / num_points

    # convert the tensors to numpy arrays for matplotlib
    grid, obj_values, satisfactory_points = grid.numpy(), obj_values.numpy(), satisfactory_points.numpy()

    # setup plot grid
    num_obj = len(objectives)
    num_plots = int(num_obj * (num_obj - 1) / 2)
    num_rows = (num_plots - 1) // 3 + 1

    # create the axs object as a 2D numpy array
    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    if num_rows == 1:
        axs: np.ndarray = np.array([axs])

    idx = 0
    for i, j in itertools.combinations(range(num_obj), 2):
        ax = axs[idx // 3, idx % 3]

        # draw the heatmap on a simplified grid
        ax.hexbin(obj_values[:, i], obj_values[:, j], mincnt=1)

        # draw the satisfaction thresholds as horizontal / vertical lines
        ax.axhline(objectives[j].abs_threshold, color="#a0a0a0", linestyle="--", lw=0.5)
        ax.axvline(objectives[i].abs_threshold, color="#a0a0a0", linestyle="--", lw=0.5)

        # draw a rectangle around the satisfied region
        try:
            if objectives[i].best_value > objectives[i].worst_value:
                satisfaction_idx_i = obj_values[:, i] >= objectives[i].abs_threshold
            else:
                satisfaction_idx_i = obj_values[:, i] <= objectives[i].abs_threshold

            if objectives[j].best_value > objectives[j].worst_value:
                satisfaction_idx_j = obj_values[:, j] >= objectives[j].abs_threshold
            else:
                satisfaction_idx_j = obj_values[:, j] <= objectives[j].abs_threshold

            satisfaction_idx = np.logical_and(satisfaction_idx_i, satisfaction_idx_j)

            if objectives[i].best_value > objectives[i].worst_value:
                best_value_i = np.max(obj_values[satisfaction_idx, i])
            else:
                best_value_i = np.min(obj_values[satisfaction_idx, i])

            if objectives[j].best_value > objectives[j].worst_value:
                best_value_j = np.max(obj_values[satisfaction_idx, j])
            else:
                best_value_j = np.min(obj_values[satisfaction_idx, j])

            rect = patches.Rectangle(
                (objectives[i].abs_threshold, objectives[j].abs_threshold),
                best_value_i - objectives[i].abs_threshold,
                best_value_j - objectives[j].abs_threshold,
                linewidth=2.,
                edgecolor="black",
                facecolor="none",
                zorder=10
            )
            ax.add_patch(rect)

        except ValueError:
            pass

        # scatter the satisfactory points in red on top of the heatmap
        ax.scatter(satisfactory_points[:, i], satisfactory_points[:, j], color="red", s=0.5, zorder=9, alpha=0.5)

        ax.set_xlabel(f"Objective {i + 1}")
        ax.set_ylabel(f"Objective {j + 1}")

        idx += 1

    # remove empty plots
    while idx < 3 * num_rows:
        axs[idx // 3, idx % 3].axis("off")
        idx += 1

    # save the plot
    fig.tight_layout()

    # file_name = file_name.replace(".png", f"_{100 * satisfaction_fraction:.2f}%.png")

    plt.savefig(file_name, dpi=600, transparent=True)
