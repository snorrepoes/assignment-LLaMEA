import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.exceptions import BotorchTensorDimensionError
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0] if initial_points is not None else 0
        self.calls_made = 0

        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Initialize best solution tracking
        self.best_f = np.inf
        self.best_x = None

        # Bounds for optimization
        self.bounds = torch.tensor([[-5.0]*dim, [5.0]*dim], dtype=torch.double)

    def __call__(self, func):
        # Evaluate initial points first
        if self.initial_points is not None:
            for x in self.initial_points:
                if self.calls_made >= self.budget:
                    raise RuntimeError("Budget exceeded during initial points evaluation")

                f = func(x)
                self.calls_made += 1

                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()

        # Continue with Bayesian optimization
        while self.calls_made < self.budget:
            # Convert current best to tensor
            best_f_tensor = torch.tensor([self.best_f], dtype=torch.double)

            # Create training data
            train_X = torch.tensor(self.initial_points[:self.calls_made - self.n_init], dtype=torch.double)
            train_Y = torch.tensor([func(x) for x in self.initial_points[:self.calls_made - self.n_init]], dtype=torch.double).unsqueeze(-1)

            # Fit GP model
            gp = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=ScaleKernel(
                    base_kernel=MaternKernel(
                        nu=2.5,
                        ard_num_dims=self.dim,
                        lengthscale_constraint=Interval(0.1, 10.0)
                    )
                )
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.to(train_X)
            fit_gpytorch_model(mll)

            # Define acquisition function
            sampler = SobolQMCNormalSampler(num_samples=512)
            acqf = qLogNoisyExpectedImprovement(
                model=gp,
                best_f=best_f_tensor,
                sampler=sampler,
                X_baseline=train_X,
                prune_baseline=True
            )

            # Optimize acquisition function
            try:
                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=10,
                    raw_samples=100,
                    options={"batch_limit": 5, "maxiter": 200},
                )
            except BotorchTensorDimensionError:
                # Fallback to random sampling if optimization fails
                candidates = torch.rand(1, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            # Evaluate candidate
            x_candidate = candidates.detach().numpy()[0]
            f_candidate = func(x_candidate)
            self.calls_made += 1

            # Update best solution
            if f_candidate < self.best_f:
                self.best_f = f_candidate
                self.best_x = x_candidate.copy()

        return self.best_f, self.best_x