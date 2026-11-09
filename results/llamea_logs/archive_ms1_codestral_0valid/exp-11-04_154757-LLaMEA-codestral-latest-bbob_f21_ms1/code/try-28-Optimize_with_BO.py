import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.transforms import Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.utils.transforms import unnormalize, normalize
import warnings

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.best_f = np.inf
        self.best_x = None
        self.X = None
        self.Y = None

    def __call__(self, func):
        self.X = torch.tensor(self.initial_points, dtype=torch.double, device=self.device)
        self.Y = torch.tensor([func(x.cpu().numpy()) for x in self.X], dtype=torch.double, device=self.device).view(-1, 1)

        self.best_f = self.Y.min().item()
        self.best_x = self.X[self.Y.argmin()].cpu().numpy()

        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double, device=self.device)

        for i in range(self.n_init, self.budget):
            if i >= self.budget:
                break

            model = self._fit_gp_model(self.X, self.Y, bounds)

            sampler = SobolQMCNormalSampler(num_samples=512)
            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=self.X,
                sampler=sampler,
            )

            candidates, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )

            x_new = candidates.detach().cpu().numpy()
            y_new = torch.tensor([func(x_new)], dtype=torch.double, device=self.device).view(-1, 1)

            self.X = torch.cat([self.X, candidates])
            self.Y = torch.cat([self.Y, y_new])

            if y_new.item() < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

        return self.best_f, self.best_x

    def _fit_gp_model(self, X, Y, bounds):
        transform = Standardize(m=X.shape[-1])
        Y_std = transform(Y)

        kernel = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=X.shape[-1],
            ),
        )

        model = SingleTaskGP(
            X,
            Y_std,
            input_transform=transform,
            covar_module=kernel,
            outcome_transform=Standardize(1),
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(self.device)
        fit_gpytorch_mll(mll)

        return model