import numpy as np
import pandas as pd
import torch
from botorch.test_functions.multi_objective import ZDT1, ZDT2, ZDT3
from botorch.utils.multi_objective import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.config import Config

Config.show_compile_hint = False
from botorch.utils.transforms import normalize, unnormalize

from jes.utils.bo_loop import bo_loop

d = 30
M = 2
n = min(60, 6 * d)
n_iteration = 3

problem = ZDT1(dim=d, num_objectives=M, noise_std=0, negate=True)
bounds = problem.bounds
standard_bounds = torch.zeros(2, d)
standard_bounds[1] = 1.0

# `n x d`
# train_X = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=123).squeeze(-2)
train_X = (bounds[1, :] - bounds[0, :]) * torch.rand(n, d) + bounds[0, :]
train_Y = problem(train_X)

hv_exact = []
for i in range(n_iteration):
    x = bo_loop(
        train_X=normalize(train_X, bounds),
        train_Y=train_Y,
        num_outputs=M,
        bounds=standard_bounds,
        acquisition_type="jes_lb",
        num_pareto_samples=10,
        num_pareto_points=10,
        num_greedy=10,
        num_samples=128,
        num_restarts=10,
        raw_samples=1000,
        batch_size=1,
    )
    x = unnormalize(x, bounds)
    y = problem(x)
    train_X = torch.cat((train_X, x), 0)
    train_Y = torch.cat((train_Y, y), 0)

    idx = is_non_dominated(train_Y)
    pf = train_Y[idx, :]
    hv = Hypervolume(problem.ref_point)
    hv_exact.append(hv.compute(pf.squeeze(0)))

N = train_X.shape[0]
df = pd.DataFrame(
    np.c_[
        [0] * n + list(range(1, n_iteration + 1)),
        train_X.numpy(),
        train_Y.numpy(),
        [0] * N,
        [0] * N,
        [0] * N,
        [0] * N,
        [0] * N,
        [0] * N,
        [0] * n + hv_exact,
    ],
    columns=["iterID"]
    + [f"x{i+1}" for i in range(d)]
    + [
        "f1",
        "f2",
        "Expected_f1",
        "Uncertainty_f1",
        "Acquisition_f1",
        "Expected_f2",
        "Uncertainty_f2",
        "Acquisition_f2",
        "Hypervolume_indicator",
    ],
)
df.to_csv("ZDT1-JES.csv")
