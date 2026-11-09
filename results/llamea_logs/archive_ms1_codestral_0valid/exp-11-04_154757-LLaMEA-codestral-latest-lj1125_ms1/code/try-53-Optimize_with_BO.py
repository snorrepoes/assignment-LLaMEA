import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.f_opt = np.inf
        self.x_opt = None
        self.evaluated_points = []
        self.evaluated_values = []

    def __call__(self, func):
        # Evaluate initial points
        for x in self.initial_points:
            f = func(x)
            self.evaluated_values.append(f)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()

        train_x = torch.tensor(np.array(self.evaluated_points), dtype=torch.float64)
        train_y = torch.tensor(np.array(self.evaluated_values), dtype=torch.float64).unsqueeze(-1)
        train_y_log = torch.log(torch.abs(train_y) + 1e-8)

        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.float64)
        train_x_normalized = normalize(train_x, bounds)

        remaining_budget = self.budget - len(self.initial_points)

        for _ in range(remaining_budget):
            # Create and fit GP model
            class ExactGPModel(ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = torch.nn.Identity()
                    self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return torch.distributions.MultivariateNormal(mean_x, covar_x)

            likelihood = GaussianLikelihood()
            model = ExactGPModel(train_x_normalized, train_y_log, likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Create acquisition function
            acqf = qLogNoisyExpectedImprovement(
                model=model,
                best_f=train_y_log.min(),
                X_baseline=train_x_normalized,
                prune_baseline=True
            )

            # Optimize acquisition function
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )

            # Evaluate candidate
            candidate = candidates.detach().squeeze().numpy()
            candidate_denormalized = bounds[0] + candidate * (bounds[1] - bounds[0])
            f = func(candidate_denormalized)

            # Update best and training data
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = candidate_denormalized.copy()

            self.evaluated_points.append(candidate_denormalized)
            self.evaluated_values.append(f)

            train_x = torch.tensor(np.array(self.evaluated_points), dtype=torch.float64)
            train_y = torch.tensor(np.array(self.evaluated_values), dtype=torch.float64).unsqueeze(-1)
            train_y_log = torch.log(torch.abs(train_y) + 1e-8)
            train_x_normalized = normalize(train_x, bounds)

        return self.f_opt, self.x_opt

def fit_gpytorch_model(mll):
    from torch.optim import Adam
    optimizer = Adam(mll.parameters(), lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        output = mll.model(mll.model.train_inputs[0])
        loss = -mll(output, mll.model.train_targets)
        loss.backward()
        optimizer.step()