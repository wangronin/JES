import torch

from problems.botorch_wrapper import RE, WOSGZ, ZDT

f = ZDT(id=1, dim=10, negate=False)
print(f.bounds)
print(f.ref_point)
print(f(torch.rand(5, 10)))

f = WOSGZ(id=2, dim=10, negate=True)
bounds = f.bounds
X = (bounds[1, :] - bounds[0, :]) * torch.rand(5, 10) + bounds[0, :]
print(f.bounds)
print(f.ref_point)
print(f(X))

f = RE(negate=True)
bounds = f.bounds
X = (bounds[1, :] - bounds[0, :]) * torch.rand(5, f.dim) + bounds[0, :]
print(f.bounds)
print(f.ref_point)
print(f(X))
