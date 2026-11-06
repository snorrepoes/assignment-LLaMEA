import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.transforms import Normalize, Standardize

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([[-5.0] * self.dim, [5.0] * self.dim], dtype=torch.double)
        X = torch.tensor(self.initial_points, dtype=torch.double)
        Y = torch.tensor([func(x) for x in X], dtype=torch.double).unsqueeze(-1)
        
        n_evals = len(Y)
        if n_evals > self.budget:
            raise ValueError("Budget exceeded")

        while n_evals < self.budget:
            # Standardize and log-transform outcomes to handle the flat plateau vs deep well issue
            train_Y = torch.log(Y - Y.min() + 1e-12)
            
            model = SingleTaskGP(
                X, train_Y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
            qNEI = qNoisyExpectedImprovement(
                model=model,
                X_baseline=X,
                sampler=sampler,
            )
            
            new_x, _ = optimize_acqf(
                qNEI, bounds=bounds, q=1, num_restarts=10, raw_samples=512
            )
            
            new_y = func(new_x.detach().numpy().flatten())
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, torch.tensor([[new_y]], dtype=torch.double)])
            n_evals += 1
            
        best_idx = torch.argmin(Y)
        return Y[best_idx].item(), X[best_idx].numpy()