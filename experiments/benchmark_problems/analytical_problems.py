from botorch.test_functions.multi_objective import BNH, CarSideImpact, DH4, DTLZ5, ZDT1
from botier import AuxiliaryObjective

# in each of these problems `XY` refers to the original multi-objective problem from botorch, and `XY_mod` describes a
# modified version with a new secondary objective (only over experiment inputs)

analytical_problems = {
    "BNH": {
        "surface": BNH(negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-140.0, abs_threshold=-60.0, output_index=0),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-50.0, abs_threshold=-11.0, output_index=1),
        ],
    },
    "BNH_mod": {
        "surface": BNH(negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-140.0, abs_threshold=-60.0, output_index=0),
            AuxiliaryObjective(
                maximize=True,
                calculation=lambda y, x: x[..., 0] - x[..., 1],
                upper_bound=BNH(negate=True).bounds[1][0] - BNH(negate=True).bounds[0][1],
                lower_bound=BNH(negate=True).bounds[0][0] - BNH(negate=True).bounds[1][1],
                abs_threshold=2.0,
            ),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-50.0, abs_threshold=-15.0, output_index=1),
        ],
    },
    "DH4": {
        "surface": DH4(dim=6, negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.0, abs_threshold=-0.15, output_index=0),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-400.0, abs_threshold=-15.0, output_index=1),
        ],
    },
    "DH4_mod": {
        "surface": DH4(dim=6, negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.0, abs_threshold=-0.15, output_index=0),
            AuxiliaryObjective(
                maximize=True,
                calculation=lambda y, x: x[..., 1],
                upper_bound=DH4(dim=6, negate=True).bounds[1][1],
                lower_bound=DH4(dim=6, negate=True).bounds[0][1],
                abs_threshold=0.6,
            ),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-400.0, abs_threshold=-15.0, output_index=1),
        ],
    },
    "DTLZ5": {
        "surface": DTLZ5(dim=4, negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.75, abs_threshold=-0.5, output_index=0),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.75, abs_threshold=-0.95, output_index=1),
        ],
    },
    "DTLZ5_mod": {
        "surface": DTLZ5(dim=4, negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.75, abs_threshold=-0.5, output_index=0),
            AuxiliaryObjective(
                maximize=False,
                calculation=lambda y, x: x[..., 2] + x[..., 3],
                lower_bound=DTLZ5(dim=4, negate=True).bounds[0][2] + DTLZ5(dim=4, negate=True).bounds[0][3],
                upper_bound=DTLZ5(dim=4, negate=True).bounds[1][2] + DTLZ5(dim=4, negate=True).bounds[1][3],
                abs_threshold=1.0
            ),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.75, abs_threshold=-0.95, output_index=1),
        ],
    },
    "ZDT1": {
        "surface": ZDT1(dim=10, negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.0, abs_threshold=-0.18, output_index=0),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-10.0, abs_threshold=-2.5, output_index=1),
        ],
    },
    "ZDT1_mod": {
        "surface": ZDT1(dim=10, negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-1.0, abs_threshold=-0.18, output_index=0),
            AuxiliaryObjective(
                maximize=False,
                calculation=lambda y, x: x[..., 1] + x[..., 5],
                lower_bound=ZDT1(dim=10, negate=True).bounds[0][1] + ZDT1(dim=10, negate=True).bounds[0][5],
                upper_bound=ZDT1(dim=10, negate=True).bounds[1][1] + ZDT1(dim=10, negate=True).bounds[1][5],
                abs_threshold=0.5
            ),
            AuxiliaryObjective(maximize=True, upper_bound=0.0, lower_bound=-10.0, abs_threshold=-2.5, output_index=1),
        ],
    },
    "CarSideImpact": {
        "surface": CarSideImpact(negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=-18.0, lower_bound=-45.0, abs_threshold=-25.0, output_index=0),
            AuxiliaryObjective(maximize=True, upper_bound=-3.5, lower_bound=-4.5, abs_threshold=-3.9, output_index=1),
            AuxiliaryObjective(maximize=True, upper_bound=-10.0, lower_bound=-14.0, abs_threshold=-12.0, output_index=2)
        ],
    },
    "CarSideImpact_mod": {
        "surface": CarSideImpact(negate=True),
        "objectives": [
            AuxiliaryObjective(maximize=True, upper_bound=-18.0, lower_bound=-45.0, abs_threshold=-25.0, output_index=0),
            AuxiliaryObjective(
                maximize=False,
                calculation=lambda y, x: x.sum(dim=-1),
                lower_bound=CarSideImpact(negate=True).bounds[0].sum(),
                upper_bound=CarSideImpact(negate=True).bounds[1].sum(),
                abs_threshold=6.0,
            ),
            AuxiliaryObjective(maximize=True, upper_bound=-3.5, lower_bound=-4.5, abs_threshold=-3.9, output_index=1),
            AuxiliaryObjective(maximize=True, upper_bound=-10.0, lower_bound=-14.0, abs_threshold=-12.0, output_index=2)
        ],
    },
}