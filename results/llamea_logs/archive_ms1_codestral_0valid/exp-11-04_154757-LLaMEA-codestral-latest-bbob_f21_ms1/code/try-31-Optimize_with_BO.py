import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
import warnings

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.n_init = len(initial_points)

        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Initialize best found
        self.best_f = np.inf
        self.best_x = None
        self.calls_made = 0

        # Track evaluations
        self.X = torch.from_numpy(initial_points).float()
        self.Y = torch.empty((0, 1))

        # BO parameters
        self.restart_interval = max(10, budget // 10)
        self.n_restarts = 3
        self.n_candidates = 1000

    def __call__(self, func):
        # Evaluate initial points
        for x in self.initial_points:
            f = func(x)
            self._update_best(f, x)
            self.Y = torch.cat([self.Y, torch.tensor([[f]])])
            self.calls_made += 1

        # Main BO loop
        while self.calls_made < self.budget:
            # Fit GP model
            train_X = self.X[:self.calls_made]
            train_Y = self.Y[:self.calls_made]

            # Use Matern kernel with nu=2.5 for multimodal landscapes
            kernel = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0)
                ),
                outputscale_prior=GammaPrior(2.0, 0.15)
            )

            gp = SingleTaskGP(train_X, train_Y, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

            # Correct GP fitting method
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from gpytorch import settings
            with settings.max_cholesky_size(1000):
                mll.train()
                optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
                for _ in range(100):
                    optimizer.zero_grad()
                    output = gp(train_X)
                    loss = -mll(output, train_Y.squeeze())
                    loss.backward()
                    optimizer.step()

            # Acquisition function with noise handling
            sampler = SobolQMCNormalSampler(num_samples=256)
            acqf = qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=train_X,
                sampler=sampler,
                prune_baseline=True
            )

            # Optimize acquisition function with restarts
            bounds = torch.tensor([func.bounds.lb, func.bounds.ub]).T.float()
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=self.n_restarts,
                raw_samples=self.n_candidates,
                options={"batch_limit": 5, "maxiter": 200}
            )

            # Evaluate candidate
            x_candidate = candidates.detach().numpy().flatten()
            f_candidate = func(x_candidate)
            self._update_best(f_candidate, x_candidate)

            # Update data
            self.X = torch.cat([self.X, torch.from_numpy(x_candidate).unsqueeze(0).float()])
            self.Y = torch.cat([self.Y, torch.tensor([[f_candidate]])])
            self.calls_made += 1

            # Periodic restart for multimodal landscapes
            if (self.calls_made - self.n_init) % self.restart_interval == 0:
                self._restart_optimization()

        return self.best_f, self.best_x

    def _update_best(self, f, x):
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()

    def _restart_optimization(self):
        # Keep top 20% of points and restart with new random points
        n_keep = max(1, int(0.2 * self.calls_made))
        if n_keep < self.calls_made:
            indices = torch.topk(self.Y.squeeze(), n_keep, largest=False).indices
            self.X = self.X[indices]
            self.Y = self.Y[indices]
            self.calls_made = n_keep