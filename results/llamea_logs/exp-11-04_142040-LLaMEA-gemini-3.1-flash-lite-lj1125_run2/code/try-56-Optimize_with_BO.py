import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.generation import gen_candidates_scipy
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.utils import fix_features

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
        
        length = 0.5 * (ub - lb)
        best_x = train_x[np.argmin(train_y)]
        best_val = min(train_y)
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
            # Log transform to handle flat plateaus and high dynamic range
            Y_log = torch.log(Y - Y.min() + 1e-12)
            
            model = SingleTaskGP(X, Y_log, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Define Trust Region
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            # Thompson Sampling within Trust Region
            sobol = draw_sobol_samples(bounds=torch.stack([tr_lb, tr_ub]), n=1024, q=1).squeeze(1)
            with torch.no_grad():
                posterior = model.posterior(sobol)
                samples = posterior.rsample(sample_shape=torch.Size([10]))
            
            # Select best candidate from Thompson samples
            best_idx = torch.argmin(samples.view(10, -1).mean(dim=0))
            new_x = sobol[best_idx]
            
            new_f = float(func(new_x.numpy()))
            train_x.append(new_x)
            train_y.append(new_f)
            
            # Adaptive Trust Region update
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                length = torch.clamp(length * 1.1, 0.05 * (ub - lb), 1.0 * (ub - lb))
            else:
                length = torch.clamp(length * 0.9, 0.05 * (ub - lb), 1.0 * (ub - lb))

        best_idx = np.argmin(train_y)
        return train_y[best_idx], train_x[best_idx].numpy()