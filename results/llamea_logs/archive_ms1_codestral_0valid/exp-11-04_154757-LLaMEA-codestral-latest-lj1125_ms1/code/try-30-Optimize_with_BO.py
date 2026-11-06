import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.best_f = np.inf
        self.best_x = None
        self.X = torch.tensor(initial_points, dtype=torch.float64)
        self.y = torch.empty(len(initial_points), dtype=torch.float64)
        self.y_std = None
        self.y_mean = None

    def __call__(self, func):
        # Evaluate initial points
        for i, x in enumerate(self.initial_points):
            if i >= self.budget:
                raise RuntimeError("Initial points exceed budget")
            y_val = func(x)
            self.y[i] = y_val
            if y_val < self.best_f:
                self.best_f = y_val
                self.best_x = x.copy()

        # Standardize the objective values (log transformation for wide range)
        self.y_std = self.y.std()
        self.y_mean = self.y.mean()
        y_transformed = torch.log(torch.abs(self.y) + 1e-8)  # Add small constant to avoid log(0)

        # Main BO loop
        for i in range(len(self.initial_points), self.budget):
            # Fit GP model
            train_X = standardize(self.X)
            train_y = (y_transformed - y_transformed.mean()) / y_transformed.std()

            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_X, train_y.unsqueeze(-1), covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(train_X)
            model.to(train_X)
            fit_gpytorch_mll(mll)

            # Define acquisition function
            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                sampler=torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True),
                objective=None,
                prune_baseline=True,
                cache_root=False,
            )

            # Optimize acquisition function
            bounds = torch.tensor([[-5.0] * self.dim, [5.0] * self.dim], dtype=torch.float64)
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )

            # Evaluate new point
            x_new = candidates.detach().numpy()[0]
            y_new = func(x_new)
            y_transformed_new = torch.log(torch.abs(torch.tensor(y_new)) + 1e-8)

            # Update best solution
            if y_new < self.best_f:
                self.best_f = y_new
                self.best_x = x_new.copy()

            # Update dataset
            self.X = torch.cat([self.X, torch.tensor(x_new).unsqueeze(0)])
            self.y = torch.cat([self.y, torch.tensor([y_new])])
            y_transformed = torch.cat([y_transformed, y_transformed_new.unsqueeze(0)])

        return self.best_f, self.best_x

def fit_gpytorch_mll(mll):
    """Fit a GPyTorch MLL using L-BFGS."""
    from torch import optim
    optimizer = optim.LBFGS(mll.parameters(), lr=1)
    def closure():
        optimizer.zero_grad()
        output = mll.model(*mll.model.train_inputs)
        loss = -mll(output, mll.model.train_targets)
        loss.backward()
        return loss
    optimizer.step(closure)