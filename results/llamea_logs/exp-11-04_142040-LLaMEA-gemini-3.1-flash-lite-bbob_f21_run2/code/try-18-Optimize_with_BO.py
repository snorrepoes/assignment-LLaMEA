import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import normalize, unnormalize

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double)
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        if num_calls > self.budget:
            raise ValueError("Initial points exceed budget.")

        while num_calls < self.budget:
            # Normalize inputs to [0, 1] for better GP performance
            train_x_norm = normalize(train_x, bounds)
            
            # Use Matern 2.5 kernel with ARD
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            fit_gpytorch_mll(mll)

            # Define Log-Expected Improvement (maximization)
            # Minimize func -> Maximize -func. We pass best_f as -min(train_y)
            qLEI = qLogExpectedImprovement(model, best_f=-torch.min(train_y))

            candidate_norm, _ = optimize_acqf(
                acq_function=qLEI,
                bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()