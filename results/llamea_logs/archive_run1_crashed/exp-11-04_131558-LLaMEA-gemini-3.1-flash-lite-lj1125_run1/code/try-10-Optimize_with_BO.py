import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        torch.set_default_dtype(torch.float64)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.float64)
        ub = torch.tensor(func.bounds.ub, dtype=torch.float64)
        
        train_x_list = []
        train_y_list = []
        
        def evaluate(x):
            if len(train_x_list) >= self.budget:
                raise Exception("Budget exhausted")
            val = float(func(x))
            train_x_list.append(x)
            train_y_list.append(val)
            return val

        for p in self.initial_points:
            evaluate(p)
            
        while len(train_x_list) < self.budget:
            train_x = torch.tensor(np.array(train_x_list), dtype=torch.float64)
            train_y = torch.tensor(np.array(train_y_list), dtype=torch.float64).unsqueeze(-1)
            
            # Use Log-transform to handle flat plateaus and deep wells
            y_min = train_y.min()
            y_transformed = torch.log(train_y - y_min + 1e-12)
            
            # Robust model initialization
            model = SingleTaskGP(train_x, y_transformed, likelihood=GaussianLikelihood(noise_constraint=None))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                pass # Fallback to default if fit fails
            
            acq = qLogNoisyExpectedImprovement(model, X_baseline=train_x)
            
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=15,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200}
            )
            
            evaluate(new_x.detach().numpy().flatten())
            
        best_idx = np.argmin(train_y_list)
        return train_y_list[best_idx], train_x_list[best_idx]