import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from typing import Callable, Tuple

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Initialize optimization state
        self.X = torch.from_numpy(initial_points).float().to(self.device)
        self.Y = torch.empty((0, 1), device=self.device)
        self.best_f = float('inf')
        self.best_x = None
        self.calls = 0

    def __call__(self, func: Callable) -> Tuple[float, np.ndarray]:
        # Evaluate initial points
        for x in self.initial_points:
            f = func(x)
            self._update_state(x, f)

        # Bayesian optimization loop
        while self.calls < self.budget:
            # Fit GP model with log-transformed objective
            train_Y = torch.log(torch.abs(self.Y) + 1e-8)
            gp = self._fit_gp(self.X, train_Y)

            # Define acquisition function (SAWEI-like approach)
            acqf = self._get_acquisition(gp, func)

            # Optimize acquisition function
            candidates, _ = optimize_acqf(
                acqf,
                bounds=torch.tensor([func.bounds.lb, func.bounds.ub], device=self.device),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

            # Evaluate candidate
            x_candidate = candidates.detach().cpu().numpy().flatten()
            f_candidate = func(x_candidate)
            self._update_state(x_candidate, f_candidate)

        return self.best_f, self.best_x

    def _fit_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        # Define kernel with lengthscale bounds
        base_kernel = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.dim,
                lengthscale_prior=None,
                lengthscale_constraint=torch.distributions.constraints.Interval(0.1, 10.0),
            ),
            outputscale_prior=None,
            outputscale_constraint=torch.distributions.constraints.Interval(0.1, 100.0),
        )

        # Create GP model with standardization
        gp = SingleTaskGP(
            X,
            Y,
            covar_module=base_kernel,
            outcome_transform=Standardize(m=self.dim),
            input_transform=Normalize(self.dim, bounds=torch.tensor([[-5.0], [5.0]], device=self.device)),
        )

        # Fit GP using gpytorch's fit_gpytorch_model
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        return gp

    def _get_acquisition(self, gp: SingleTaskGP, func: Callable) -> qLogExpectedImprovement:
        # Get current best in log space
        best_Y = torch.log(torch.abs(torch.tensor(self.best_f, device=self.device)) + 1e-8)

        # Create acquisition function with exploration bonus
        acqf = qLogExpectedImprovement(
            model=gp,
            best_f=best_Y,
            maximize=False,
        )

        return acqf

    def _update_state(self, x: np.ndarray, f: float) -> None:
        self.calls += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()

        # Update GP data
        self.X = torch.cat([self.X, torch.from_numpy(x.reshape(1, -1)).float().to(self.device)])
        self.Y = torch.cat([self.Y, torch.tensor([[f]], device=self.device)])

        if self.calls > self.budget:
            raise RuntimeError(f"Budget exceeded: {self.calls} > {self.budget}")

# Helper function to fit GP model (replaces botorch.fit.fit_gpytorch_model)
def fit_gpytorch_model(mll: MarginalLogLikelihood):
    from gpytorch import settings
    from torch import optim

    with settings.max_cholesky_size(10000):
        optimizer = optim.Adam(mll.parameters(), lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            output = mll(mll.model.train_targets, mll.model(mll.model.train_inputs[0]))
            loss = -output
            loss.backward()
            optimizer.step()