import numpy as np
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

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
        bounds = torch.stack([lb, ub])
        
        X_list = []
        Y_list = []
        
        def evaluate(x):
            if len(X_list) >= self.budget:
                return
            val = func(x)
            X_list.append(torch.tensor(x, dtype=torch.double))
            Y_list.append(val)

        for x_init in self.initial_points:
            evaluate(x_init)

        while len(X_list) < self.budget:
            train_X = torch.stack(X_list)
            # Use negative log for minimization as we are looking for the deepest well
            train_Y = -torch.tensor(Y_list, dtype=torch.double).unsqueeze(-1)
            
            # Use a robust GP with Matern 5/2 kernel
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Use qNoisyExpectedImprovement which is robust to noise and flat landscapes
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
            acq_func = qNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                sampler=sampler,
            )
            
            candidates, _ = optimize_acqf(
                acq_func, 
                bounds=bounds, 
                q=1, 
                num_restarts=20, 
                raw_samples=1024,
            )
            
            evaluate(candidates.detach().numpy().flatten())

        best_idx = np.argmin(Y_list)
        return float(Y_list[best_idx]), X_list[best_idx].numpy()