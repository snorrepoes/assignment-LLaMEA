import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.generation import MaxPosteriorSampling

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
        
        train_x = [torch.tensor(x, dtype=torch.double) for x in self.initial_points]
        train_y = [float(func(x)) for x in self.initial_points]
        
        length = 1.6
        length_min, length_max = 0.01, 2.0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform to handle flat plateaus and high dynamic range
            Y_transformed = torch.log(Y - Y.min() + 1e-9)
            
            model = SingleTaskGP(
                X, Y_transformed,
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = np.argmin(train_y)
            best_x = X[best_idx]
            
            # Use Thompson Sampling for better exploration in the trust region
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            new_x = sampler(torch.stack([tr_lb, tr_ub]), num_samples=512)
            new_x_val = new_x.detach().squeeze(0)
            
            new_f = float(func(new_x_val.numpy()))
            
            # Adaptive trust region adjustment
            if new_f < train_y[best_idx]:
                length = min(length * 1.1, length_max)
            else:
                length = max(length * 0.7, length_min)
                # Restart logic if stuck in plateau
                if len(train_x) % 10 == 0:
                    length = 1.6
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        idx = np.argmin(train_y)
        return float(train_y[idx]), train_x[idx].numpy()