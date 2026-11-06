"""Vanilla BoTorch BO baseline (GP + LogEI). Surrogate/acquisition settings are constructor args
so they can be set to whatever the instructors specify. Follows BoTorch tutorial patterns
(https://botorch.org/docs/tutorials/)."""
import numpy as np, torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition="logei", matern_nu=2.5,
                 num_restarts=8, raw_samples=128):
        self.budget, self.dim, self.seed = budget, dim, random_seed
        self.X0 = np.asarray(initial_points, float)
        self.acq, self.nu, self.nr, self.rs = acquisition, matern_nu, num_restarts, raw_samples

    def __call__(self, func):
        torch.manual_seed(self.seed)
        lb, ub = func.bounds.lb, func.bounds.ub
        X = [x for x in self.X0[: self.budget]]
        y = [func(x) for x in X]
        bounds = torch.tensor(np.stack([lb, ub]), dtype=torch.double)
        while len(y) < self.budget:
            Xt = torch.tensor(np.array(X), dtype=torch.double)
            yt = -torch.tensor(y, dtype=torch.double).unsqueeze(-1)   # BoTorch maximises
            gp = SingleTaskGP(Xt, yt, covar_module=ScaleKernel(MaternKernel(nu=self.nu, ard_num_dims=self.dim)),
                              input_transform=Normalize(d=self.dim, bounds=bounds), outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acq = (LogExpectedImprovement(gp, best_f=yt.max()) if self.acq == "logei"
                   else UpperConfidenceBound(gp, beta=2.0))
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=self.nr, raw_samples=self.rs)
            x = cand.squeeze(0).numpy(); X.append(x); y.append(func(x))
        i = int(np.argmin(y))
        return y[i], X[i]
