import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=float)
        if self.initial_points.ndim != 2 or self.initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        self.random_seed = int(random_seed)

        torch.set_default_dtype(torch.double)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self._num_calls = 0
        self._X = []
        self._Y = []
        self._best_f = np.inf
        self._best_x = None

        self._lb = None
        self._ub = None
        self._lb_tensor = None
        self._ub_tensor = None
        self._bounds_tensor = None

    def _evaluate(self, func, x):
        if self._num_calls >= self.budget:
            raise RuntimeError("Evaluation budget exceeded")
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError("x must have length dim")
        y = float(func(x))
        self._num_calls += 1
        self._X.append(x)
        self._Y.append(y)
        if y < self._best_f:
            self._best_f = y
            self._best_x = x.copy()
        return y

    def _random_point(self):
        g = torch.Generator().manual_seed(self.random_seed + self._num_calls * 1000)
        return (
            torch.rand(self.dim, dtype=torch.double, generator=g)
            * (self._ub_tensor - self._lb_tensor)
            + self._lb_tensor
        ).cpu().numpy()

    def _fit_gp(self):
        X = torch.tensor(np.array(self._X), dtype=torch.double)
        Y = torch.tensor(np.array(self._Y), dtype=torch.double).unsqueeze(-1)

        likelihood = GaussianLikelihood()
        model = SingleTaskGP(
            X,
            Y,
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
            likelihood=likelihood,
            input_transform=Normalize(d=self.dim, bounds=self._bounds_tensor),
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll, max_attempts=10)
        except Exception:
            pass

        model.eval()
        return model

    def __call__(self, func):
        lb = np.broadcast_to(np.asarray(func.bounds.lb, dtype=float), (self.dim,)).copy()
        ub = np.broadcast_to(np.asarray(func.bounds.ub, dtype=float), (self.dim,)).copy()
        if np.any(lb >= ub):
            raise ValueError("Invalid bounds: lb must be smaller than ub")

        self._lb = lb
        self._ub = ub
        self._lb_tensor = torch.tensor(lb, dtype=torch.double)
        self._ub_tensor = torch.tensor(ub, dtype=torch.double)
        self._bounds_tensor = torch.stack([self._lb_tensor, self._ub_tensor], dim=0)

        if len(self.initial_points) > self.budget:
            raise RuntimeError("initial_points exceed budget")

        for x in self.initial_points:
            self._evaluate(func, x)

        n_init = len(self.initial_points)

        while self._num_calls < self.budget:
            if len(self._X) < 3:
                x_next = self._random_point()
            else:
                torch.manual_seed(self.random_seed + self._num_calls * 100)
                np.random.seed(self.random_seed + self._num_calls * 100)

                model = self._fit_gp()

                # Alternate exploitation (qLogEI) and exploration (qUCB)
                if (self._num_calls - n_init) % 2 == 0:
                    acq = qLogExpectedImprovement(model=model, best_f=self._best_f)
                else:
                    acq = qUpperConfidenceBound(model=model, beta=2.0)

                try:
                    candidates, _ = optimize_acqf(
                        acq,
                        bounds=self._bounds_tensor,
                        q=1,
                        num_restarts=15,
                        raw_samples=96,
                        options={"maxiter": 150},
                    )
                    x_next = candidates[0].detach().cpu().numpy().reshape(-1)
                except Exception:
                    x_next = self._random_point()

                x_next = np.clip(x_next, self._lb, self._ub)

            self._evaluate(func, x_next)

        return float(self._best_f), self._best_x.copy()