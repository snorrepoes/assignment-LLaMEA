import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from gpytorch.mlls import fit_gpytorch_model

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0] if initial_points is not None else 0
        self.calls_used = 0
        self.best_f = np.inf
        self.best_x = None

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def __call__(self, func):
        if self.initial_points is not None:
            for x in self.initial_points:
                f = func(x)
                self.calls_used += 1
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()

        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.float64)

        while self.calls_used < self.budget:
            train_x = torch.tensor(self.initial_points if self.calls_used == self.n_init else
                                 np.vstack([self.initial_points, np.array([self.best_x])]), dtype=torch.float64)
            train_y = torch.tensor([self.best_f] * train_x.shape[0], dtype=torch.float64).unsqueeze(-1)
            train_y_transformed = torch.log(train_y + 1e-8)

            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.1, 10.0)))
            model = SingleTaskGP(train_x, train_y_transformed, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            sampler = SobolQMCNormalSampler(num_samples=1024)
            acqf = qLogExpectedImprovement(model=model, best_f=train_y_transformed.min(), sampler=sampler)

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

            x_candidate = candidates.detach().numpy().squeeze()
            f_candidate = func(x_candidate)
            self.calls_used += 1

            if f_candidate < self.best_f:
                self.best_f = f_candidate
                self.best_x = x_candidate.copy()

            self.initial_points = np.vstack([self.initial_points, x_candidate]) if self.initial_points is not None else np.array([x_candidate])

        return self.best_f, self.best_x