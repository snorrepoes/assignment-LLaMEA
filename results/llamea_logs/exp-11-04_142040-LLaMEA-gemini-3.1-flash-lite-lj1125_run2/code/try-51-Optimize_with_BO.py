import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from torch.quasirandom import SobolEngine

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
        train_y = [float(func(x)) for x in self.initial_points]

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            
            # Log transform Y to handle high dynamic range: Y = log(Y - min + eps)
            # Minimize -log(Y - min + eps) is equivalent to minimizing Y
            y_min = Y.min()
            Y_trans = -torch.log(Y - y_min + 1e-10)
            
            model = SingleTaskGP(X, Y_trans, likelihood=GaussianLikelihood())
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # TuRBO-1: Local search centered on the best observation
            best_idx = torch.argmin(Y)
            best_x = X[best_idx]
            
            # Adaptive Trust Region size (decaying)
            progress = len(train_x) / self.budget
            tr_size = 2.0 * (0.1 ** progress) 
            tr_lb = torch.clamp(best_x - tr_size, lb, ub)
            tr_ub = torch.clamp(best_x + tr_size, lb, ub)

            # Use qNEI to handle potential noise and guide exploration
            qnei = qNoisyExpectedImprovement(model, X_baseline=X, sampler=None)
            
            new_x, _ = optimize_acqf(
                acq_function=qnei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            
            val = func(new_x.detach().numpy()[0])
            train_x.append(new_x.squeeze(0))
            train_y.append(float(val))

        train_y_arr = np.array(train_y)
        best_idx = np.argmin(train_y_arr)
        return float(train_y_arr[best_idx]), train_x[best_idx].numpy()