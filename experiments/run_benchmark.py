from pathlib import Path
import argparse
import time
import warnings
from tqdm import tqdm

# append the current directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.resolve()))

# Torch GPyTorch imports
import torch
torch.set_default_dtype(torch.float64)

# Botorch imports
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim.initializers import BadInitialCandidatesWarning

# BOTier imports
from botier import HierarchyScalarizationObjective, ObjectiveCalculator

# Custom imports for benchmarking purposes
from utils.recommendation import recommend_new_point, generate_seed_data
from utils.logging_utils import get_logger
from reference_methods.penalty_scalarization import PenaltyScalarizationObjective
from reference_methods.chimera_wrapper import ChimeraWrapper
from benchmark_problems.emulated_problems import emulated_problems
from benchmark_problems.analytical_problems import analytical_problems


# annoying botorch warning that pops up if you use EI as the acquisition function
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)


results_dir = Path.cwd() / "results"
results_dir.mkdir(exist_ok=True)


if __name__ == "__main__":

    problems = emulated_problems | analytical_problems

    ####################################################################################################################
    # Argument Parsing
    ####################################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem", type=str, help="Problem to benchmark over. Must be a key in the `benchmark_problems` folder.")

    parser.add_argument("--iterations", type=int, default=50, help="Number of independent optimization campaigns.")
    parser.add_argument("--seed_budget", type=int, default=1, help="Number of seed data points to generate by random sampling.")
    parser.add_argument("--budget", type=int, default=50, help="Number of optimization steps per campaign.")
    parser.add_argument("--num_posterior_samples", type=int, default=512, help="Number of samples to draw from the posterior.")
    parser.add_argument("--k", type=float, default=1E2, help="Smoothing parameter for the HierarchyScalarizationObjective.")

    parser.add_argument("-s", "--strategy", type=str, default="botier-ei", help="Acquisition function to use.")

    parser.add_argument("--black_box_objective", action="store_true", default=False, help="Whether to train the surrogate on the score (True) or on individual objectives (False).")
    parser.add_argument("--joint_model", action="store_true", default=False, help="Whether to use a single multi-output model (True) or separate single-output models (False).")

    parser.add_argument("--restart", action="store_true", default=False, help="Whether to restart the benchmark from scratch (True) or continue from the last checkpoint (False).")

    args = parser.parse_args()

    ####################################################################################################################

    base_name = f"{args.problem}_{'black_box' if args.black_box_objective else 'composite'}_{'joint' if args.joint_model else 'multi'}_{args.strategy}_k{args.k:.0f}"
    logger = get_logger("A", logfile=results_dir / f"{base_name}.log")

    logger.info(f"Running benchmark for problem: {args.problem} with the following settings:")
    logger.info(f"    {'Black_Box' if args.black_box_objective else 'Composite'} objective")
    logger.info(f"    {'Joint Model' if args.joint_model else 'Multiple Single-Task Models'}")
    logger.info(f"    Strategy: {args.strategy}")
    logger.info(f"    k: {args.k:.0f}")
    logger.info(f"    {args.iterations} iterations with {args.budget} optimization steps each")

    surface, objectives = problems[args.problem]["surface"], problems[args.problem]["objectives"]

    acqf_options = {
        # Hierarchical Scalarization of Multiple Objectives into a Single Objective Score (using BOTier)
        # Acquisition Function: Expected Improvement
        "botier-ei": {
            "acqf_type": qExpectedImprovement,
            "acqf_kwargs": {},
            "obj_type": HierarchyScalarizationObjective,
            "obj_kwargs": {"normalized_objectives": True, "k": args.k}
        },
        # Hierarchical Scalarization of Multiple Objectives into a Single Objective Score (using BOTier)
        # Acquisition Function: Upper Confidence Bound
        "botier-ucb": {
            "acqf_type": qUpperConfidenceBound,
            "acqf_kwargs": {"beta": 0.5},
            "obj_type": HierarchyScalarizationObjective,
            "obj_kwargs": {"normalized_objectives": True, "k": args.k}
        },
        # Hierarchical Scalarization of Multiple Objectives into a Single Objective Score (using Chimera)
        # Acquisition Function: Expected Improvement
        "chimera-ei": {
            "acqf_type": qExpectedImprovement,
            "acqf_kwargs": {},
            "obj_type": ChimeraWrapper,
            "obj_kwargs": {"normalized_objectives": True, "k": args.k}
        },
        # Penalty Scalarization of Multiple Objectives into a Single Objective Score
        # Acquisition Function: Expected Improvement
        "penalty-ei": {
            "acqf_type": qExpectedImprovement,
            "acqf_kwargs": {},
            "obj_type": PenaltyScalarizationObjective,
            "obj_kwargs": {"normalized_objectives": True, "k": args.k}
        },
        # Multi-Objective Optimization Using the Expected Hypervolume Improvement
        "ehvi": {
            "acqf_type": qExpectedHypervolumeImprovement,
            "acqf_kwargs": {"ref_point": torch.zeros(len(objectives), dtype=torch.get_default_dtype())},
            "obj_type": ObjectiveCalculator,
            "obj_kwargs": {"normalized_objectives": True}
        },
        # Multi-Objective Optimization Using the Expected Hypervolume Improvement Relative to the User-Defined
        # Thresholds
        "ehvi-thresh": {
            "acqf_type": qExpectedHypervolumeImprovement,
            "acqf_kwargs": {"ref_point": torch.ones(len(objectives), dtype=torch.get_default_dtype())},
            "obj_type": ObjectiveCalculator,
            "obj_kwargs": {"normalized_objectives": True}
        },
        "sobol": {
            "acqf_type": None,
            "obj_type": ObjectiveCalculator,
            "obj_kwargs": {"normalized_objectives": True}
        }
    }

    acq_objective = acqf_options[args.strategy]["obj_type"](
        objectives=objectives,
        **acqf_options[args.strategy]["obj_kwargs"]
    )
    ref_objective = HierarchyScalarizationObjective(objectives=objectives, normalized_objectives=True, k=args.k)

    file_name = results_dir / f"{base_name}.pt"

    if args.restart is False:
        try:
            results = torch.load(file_name)
            all_x, all_y, all_obj, all_s = results["x"], results["y"], results["obj"], results["s"]
            logger.info(f"Found a previous checkpoint with {all_x.shape[0]} completed iterations. Continuing from there.")
            iteration = all_x.shape[0]
            all_x, all_y, all_obj, all_s = [x for x in all_x], [y for y in all_y], [obj for obj in all_obj], [s for s in all_s]

        except FileNotFoundError:
            logger.info("No previous checkpoint found. Starting from scratch.")
            all_x, all_y, all_obj, all_s = [], [], [], []
            iteration = 0
    else:
        logger.info("Restarting the benchmark from scratch.")
        all_x, all_y, all_obj, all_s = [], [], [], []
        iteration = 0

    ####################################################################################################################

    while iteration < args.iterations:
        start_time = time.time()

        if args.strategy == "sobol":
            x, y = generate_seed_data(surface, args.budget, random_seed=iteration)

        else:

            x, y = generate_seed_data(surface, args.seed_budget, random_seed=iteration)

            for _ in tqdm(range(args.budget), desc=f"Iteration {iteration+1}/{args.iterations}"):
                new_x = recommend_new_point(
                    x_train=x,
                    y_train=y,
                    acqf_type=acqf_options[args.strategy]["acqf_type"],
                    acqf_kwargs=acqf_options[args.strategy]["acqf_kwargs"],
                    acq_objective=acq_objective,
                    bounds=surface.bounds,
                    joint_model=args.joint_model,
                    black_box_objective=args.black_box_objective,
                    random_seed=iteration
                )
                new_y = surface(new_x)

                # check if any elements in new_x or new_y are NaN
                if torch.isnan(new_x).any() or torch.isnan(new_y).any():
                    logger.warning(f"NaNs detected in the new data (x: {new_x.flatten()}; y: {new_y.flatten()}). Skipping "
                                   f"this iteration. Using a random recommendation instead.")
                    new_x, new_y = generate_seed_data(surface, 1, random_seed=iteration)

                x = torch.cat([x, new_x], dim=0)
                y = torch.cat([y, new_y], dim=0)

        all_x.append(x), all_y.append(y)
        all_obj.append(acq_objective.calculate_objective_values(y, x, normalize=False))
        all_s.append(ref_objective(y, x))

        logger.info(f"Completed iteration {iteration+1} / {args.iterations} in {time.time() - start_time:.1f} sec. "
                    f"Saving checkpoint.")

        torch.save(
            {
                "x": torch.stack(all_x, dim=0),
                "y": torch.stack(all_y, dim=0),
                "obj": torch.stack(all_obj, dim=0),
                "s": torch.stack(all_s, dim=0)
            },
            file_name
        )

        iteration += 1

    ####################################################################################################################
