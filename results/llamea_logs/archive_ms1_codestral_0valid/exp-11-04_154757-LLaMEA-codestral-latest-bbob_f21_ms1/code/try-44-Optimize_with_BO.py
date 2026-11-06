import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import fit_gpytorch_mll

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0]
        self.best_f = np.inf
        self.best_x = None

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.bounds = torch.tensor([[-5.0] * dim, [5.0] * dim], dtype=torch.float64)

    def __call__(self, func):
        train_x = torch.tensor(self.initial_points, dtype=torch.float64)
        train_y = torch.tensor([func(x.numpy()) for x in train_x], dtype=torch.float64).unsqueeze(-1)

        self.best_f = train_y.min().item()
        self.best_x = train_x[train_y.argmin()].numpy()

        remaining_budget = self.budget - self.n_init

        for _ in range(remaining_budget):
            train_x_norm = normalize(train_x, self.bounds)
            train_y_norm = (train_y - train_y.mean()) / (train_y.std() + 1e-6)

            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, train_y_norm, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acqf = qLogExpectedImprovement(
                model=model,
                best_f=train_y_norm.min(),
                maximize=False
            )

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200}
            )

            candidate = unnormalize(candidates.detach(), self.bounds)
            candidate_np = candidate.numpy().flatten()

            f = func(candidate_np)
            f_tensor = torch.tensor([[f]], dtype=torch.float64)

            train_x = torch.cat([train_x, candidate])
            train_y = torch.cat([train_y, f_tensor])

            if f < self.best_f:
                self.best_f = f
                self.best_x = candidate_np

        return self.best_f, self.best_x