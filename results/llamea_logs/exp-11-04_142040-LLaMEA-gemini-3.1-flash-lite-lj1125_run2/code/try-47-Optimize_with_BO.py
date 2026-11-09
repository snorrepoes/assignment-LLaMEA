import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unstandardize

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
        
        best_x = train_x[np.argmin(train_y)]
        best_f = min(train_y)
        
        length = 1.6
        success_counter = 0
        failure_counter = 0

        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Log transform to manage dynamic range
            y_min = Y.min()
            Y_log = -torch.log(Y - y_min + 1e-12)
            
            model = SingleTaskGP(X, Y_log)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Thompson Sampling-like exploration using UCB
            acq_func = qUpperConfidenceBound(model, beta=2.0)
            
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            new_x, _ = optimize_acqf(
                acq_func, bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=10, raw_samples=128
            )
            
            new_f = func(new_x.detach().numpy()[0])
            train_x.append(new_x.squeeze(0))
            train_y.append(new_f)
            
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.squeeze(0)
                success_counter += 1
                failure_counter = 0
            else:
                failure_counter += 1
            
            # Adaptive trust region adjustment
            if success_counter >= 3:
                length = min(2.0, length * 2.0)
                success_counter = 0
            elif failure_counter >= 5:
                length = length / 2.0
                failure_counter = 0
                
            if length < 1e-3:
                length = 1.0 # Reset TR
        
        return float(min(train_y)), train_x[np.argmin(train_y)].numpy()