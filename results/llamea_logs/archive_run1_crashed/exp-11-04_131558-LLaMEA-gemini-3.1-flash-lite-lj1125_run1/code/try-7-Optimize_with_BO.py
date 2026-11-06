import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.float64)
        ub = torch.tensor(func.bounds.ub, dtype=torch.float64)
        
        train_x, train_y = [], []
        
        def evaluate(x):
            if len(train_x) >= self.budget:
                raise Exception("Budget exceeded")
            val = float(func(x))
            train_x.append(x)
            train_y.append(val)
            return val

        for p in self.initial_points:
            evaluate(p)
            
        while len(train_x) < self.budget:
            X = torch.tensor(np.array(train_x), dtype=torch.float64)
            y_raw = torch.tensor(np.array(train_y), dtype=torch.float64).unsqueeze(-1)
            
            # Robust log-transform with shift to handle flat plateaus and scale
            y_min = y_raw.min()
            y_transformed = torch.log(y_raw - y_min + 1e-6)
            
            # GP with noise prior to handle numerical instability
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))
            kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(X, y_transformed, likelihood=likelihood, covar_module=kernel)
            
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except:
                pass # Continue with default if fit fails
                
            acq = qNoisyExpectedImprovement(model, X_baseline=X)
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200}
            )
            
            evaluate(new_x.detach().numpy().flatten())
            
        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx]