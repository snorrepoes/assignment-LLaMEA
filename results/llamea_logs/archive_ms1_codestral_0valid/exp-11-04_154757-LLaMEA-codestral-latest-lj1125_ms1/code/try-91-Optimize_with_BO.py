import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
from typing import Callable, Tuple

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = initial_points.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.X = torch.from_numpy(initial_points).float().to(self.device)
        self.Y = torch.empty((0, 1), device=self.device)
        self.best_f = float('inf')
        self.best_x = None
        self.calls = 0

    def __call__(self, func: Callable) -> Tuple[float, np.ndarray]:
        for x in self.initial_points:
            f = func(x)
            self._update_state(x, f)

        while self.calls < self.budget:
            train_Y = torch.log(torch.abs(self.Y) + 1e-8)
            gp = self._fit_gp(self.X, train_Y)

            acqf = self._get_acquisition(gp, func)

            candidates, _ = optimize_acqf(
                acqf,
                bounds=torch.tensor([func.bounds.lb, func.bounds.ub], device=self.device),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )

            x_candidate = candidates.detach().cpu().numpy().flatten()
            f_candidate = func(x_candidate)
            self._update_state(x_candidate, f_candidate)

        return self.best_f, self.best_x

    def _fit_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
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

        gp = SingleTaskGP(
            X,
            Y,
            covar_module=base_kernel,
            outcome_transform=Standardize(m=self.dim),
            input_transform=Normalize(self.dim, bounds=torch.tensor([[-5.0], [5.0]], device=self.device)),
        )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        return gp

    def _get_acquisition(self, gp: SingleTaskGP, func: Callable) -> qLogExpectedImprovement:
        best_Y = torch.log(torch.abs(torch.tensor(self.best_f, device=self.device)) + 1e-8)

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

        self.X = torch.cat([self.X, torch.from_numpy(x.reshape(1, -1)).float().to(self.device)])
        self.Y = torch.cat([self.Y, torch.tensor([[f]], device=self.device)])

        if self.calls > self.budget:
            raise RuntimeError(f"Budget exceeded: {self.calls} > {self.budget}")