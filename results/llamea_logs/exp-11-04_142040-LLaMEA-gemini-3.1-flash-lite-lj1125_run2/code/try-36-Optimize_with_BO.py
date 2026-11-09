import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from botorch.models.transforms.outcome import LogStandardize

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        train_x = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y = [torch.tensor(func(x), dtype=torch.double) for x in self.initial_points]

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.stack(train_y).unsqueeze(-1)
            
            # Using Log-transformed GP to map the wide plateau to a manageable range
            model = SingleTaskGP(X, Y, outcome_transform=LogStandardize())
            model.likelihood.noise_prior = GammaPrior(1.1, 0.05)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # TuRBO-1: Local search using a trust region centered on current best
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Decay trust region size to narrow search to the global optimum
            tr_size = 1.0 * (0.98 ** (len(train_x) - len(self.initial_points)))
            tr_lb = torch.clamp(best_x - tr_size, lb, ub)
            tr_ub = torch.clamp(best_x + tr_size, lb, ub)

            # Use qLogEI for better exploration of high dynamic range, focusing on improvements
            acqf = qLogExpectedImprovement(model, best_f=Y.min())
            
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_val = torch.tensor(func(new_x.detach().squeeze().numpy()), dtype=torch.double)
            train_x.append(new_x.squeeze(0))
            train_y.append(new_val)

        train_y_np = torch.stack(train_y).numpy()
        best_idx = np.argmin(train_y_np)
        return float(train_y_np[best_idx]), train_x[best_idx].numpy()