from typing import Optional

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from pymoo.problems.multi import *
from torch import Tensor

from real_world import RE1
from wosgz import *


class ZDT(MultiObjectiveTestProblem):
    _ref_point = [15.0, 15.0]

    def __init__(
        self,
        id: int,
        dim: int,
        num_objectives: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
    ) -> None:
        if num_objectives != 2:
            raise NotImplementedError(f"{type(self).__name__} currently only supports 2 objectives.")
        if dim < num_objectives:
            raise ValueError(f"dim must be >= num_objectives, but got {dim} and {num_objectives}")
        self.num_objectives = num_objectives
        self.dim = dim
        self._problem = globals()[f"ZDT{id}"](n_var=self.dim)
        self._bounds = list(zip(*self._problem.bounds()))
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        F = self._problem.evaluate(X.numpy())
        return torch.from_numpy(F).to(torch.float32)

    def gen_pareto_front(self, n: int) -> Tensor:
        return torch.from_numpy(self._problem.pareto_front(n)).to(torch.float32)


class WOSGZ(MultiObjectiveTestProblem):
    _ref_point = [1.2, 1.2]

    def __init__(
        self,
        id: int,
        dim: int,
        num_objectives: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
    ) -> None:
        if num_objectives != 2:
            raise NotImplementedError(f"{type(self).__name__} currently only supports 2 objectives.")
        if dim < num_objectives:
            raise ValueError(f"dim must be >= num_objectives, but got {dim} and {num_objectives}")
        self.num_objectives = num_objectives
        self.dim = dim
        self._problem = globals()[f"WOSGZ{id}"](n_var=self.dim, n_obj=2)
        self._bounds = list(zip(*self._problem.bounds()))
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        F = self._problem.evaluate(X.numpy())
        return torch.from_numpy(F).to(torch.float32)

    def gen_pareto_front(self, n: int) -> Tensor:
        return torch.from_numpy(self._problem.pareto_front(n)).to(torch.float32)


class RE(MultiObjectiveTestProblem):
    _ref_point = [2967.0243, 0.0383]

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
    ) -> None:
        self._problem = RE1()
        self.dim = self._problem.n_var
        self.num_objectives = self._problem.n_obj
        self._bounds = list(zip(*self._problem.bounds()))
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        F = self._problem.evaluate(X.numpy())
        return torch.from_numpy(F).to(torch.float32)

    def gen_pareto_front(self, n: int) -> Tensor:
        return torch.from_numpy(self._problem.pareto_front(n)).to(torch.float32)


# f = ZDT(1, 10)
# print(f.bounds)
# print(f(torch.rand(5, 10)))

# f = WOSGZ(2, 10)
# bounds = f.bounds
# X = (bounds[1, :] - bounds[0, :]) * torch.rand(5, 10) + bounds[0, :]
# print(f.bounds)
# print(f(X))

# f = RE()
# bounds = f.bounds
# X = (bounds[1, :] - bounds[0, :]) * torch.rand(5, f.dim) + bounds[0, :]
# print(f.bounds)
# print(f(X))
