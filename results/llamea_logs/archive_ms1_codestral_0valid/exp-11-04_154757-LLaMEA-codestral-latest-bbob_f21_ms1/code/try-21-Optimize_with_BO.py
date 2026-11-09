import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.transforms import Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Initialize tracking variables
        self.best_f = np.inf
        self.best_x = None
        self.X = None
        self.Y = None

    def __call__(self, func):
        # Evaluate initial points
        self.X = torch.tensor(self.initial_points, dtype=torch.double, device=self.device)
        self.Y = torch.tensor([func(x.cpu().numpy()) for x in self.X], dtype=torch.double, device=self.device).view(-1, 1)

        # Update best found
        self.best_f = self.Y.min().item()
        self.best_x = self.X[self.Y.argmin()].cpu().numpy()

        # Define bounds
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double, device=self.device)

        # Main optimization loop
        for i in range(self.n_init, self.budget):
            if i >= self.budget:
                break

            # Fit GP model
            model = self._fit_gp_model(self.X, self.Y, bounds)

            # Define acquisition function
            acqf = self._get_acquisition_function(model, self.Y, bounds)

            # Optimize acquisition function
            candidates, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )

            # Evaluate new candidate
            x_new = candidates.detach().cpu().numpy()
            y_new = torch.tensor([func(x_new)], dtype=torch.double, device=self.device).view(-1, 1)

            # Update data
            self.X = torch.cat([self.X, candidates])
            self.Y = torch.cat([self.Y, y_new])

            # Update best found
            if y_new.item() < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

        return self.best_f, self.best_x

    def _fit_gp_model(self, X, Y, bounds):
        # Standardize the data
        transform = Standardize(m=X.shape[-1])
        Y_std = transform(Y)

        # Define kernel
        kernel = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=X.shape[-1],
                lengthscale_prior=None,
                lengthscale_constraint=None,
            ),
            outputscale_prior=None,
            outputscale_constraint=None,
        )

        # Create GP model
        model = SingleTaskGP(
            X,
            Y_std,
            input_transform=transform,
            covar_module=kernel,
            outcome_transform=Standardize(1),
        ).to(self.device)

        # Fit the model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def _get_acquisition_function(self, model, Y, bounds):
        # Use qLogNoisyExpectedImprovement for multimodal landscapes
        sampler = SobolQMCNormalSampler(num_samples=512)
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=self.X,
            sampler=sampler,
            prune_baseline=True,
        )

        return acqf