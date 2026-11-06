import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

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
        train_y = [torch.tensor(float(func(x)), dtype=torch.double) for x in self.initial_points]

        best_x = train_x[np.argmin([y.item() for y in train_y])]
        best_val = min([y.item() for y in train_y])
        
        length = 2.0
        length_min, length_max = 0.1, 4.0
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y = torch.stack(train_y).unsqueeze(-1)
            
            # Log transform to handle dynamic range and plateau
            Y_trans = -torch.log(Y - Y.min() + 1e-12)
            
            model = SingleTaskGP(
                X, Y_trans, 
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                likelihood=GaussianLikelihood()
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Trust Region constrained UCB
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qUpperConfidenceBound(model, beta=2.0)
            new_x, _ = optimize_acqf(acq, bounds=torch.stack([tr_lb, tr_ub]), q=1, 
                                     num_restarts=10, raw_samples=256)
            
            new_x = new_x.detach().flatten()
            new_f = float(func(new_x.numpy()))
            
            train_x.append(new_x)
            train_y.append(torch.tensor(new_f, dtype=torch.double))
            
            if new_f < best_val:
                best_val, best_x = new_f, new_x
                length = min(length * 1.2, length_max)
            else:
                length = max(length * 0.7, length_min)

        best_idx = np.argmin([y.item() for y in train_y])
        return train_y[best_idx].item(), train_x[best_idx].numpy()