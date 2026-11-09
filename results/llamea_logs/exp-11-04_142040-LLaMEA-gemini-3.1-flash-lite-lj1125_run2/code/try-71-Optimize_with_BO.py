import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

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
        
        length = 0.8
        min_length = 0.01
        max_length = 1.6
        
        while len(train_x) < self.budget:
            X = torch.stack(train_x)
            Y_raw = torch.tensor(train_y, dtype=torch.double).view(-1, 1)
            
            # Log transform with shift to focus on the narrow deep well
            Y_trans = torch.log(Y_raw - Y_raw.min() + 1e-9)
            
            model = SingleTaskGP(
                X, Y_trans,
                input_transform=Normalize(d=self.dim, bounds=torch.stack([lb, ub])),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, 
                                         lengthscale_prior=GammaPrior(3.0, 6.0)))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_idx = torch.argmin(Y_raw)
            best_x = X[best_idx]
            
            # Trust region logic
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
            new_x, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            
            new_x_val = new_x.detach().squeeze(0)
            new_f = float(func(new_x_val.numpy()))
            
            if new_f < Y_raw[best_idx]:
                length = min(length * 1.1, max_length)
            else:
                length = max(length * 0.7, min_length)
            
            train_x.append(new_x_val)
            train_y.append(new_f)

        final_y = np.array(train_y)
        best_idx = np.argmin(final_y)
        return float(final_y[best_idx]), train_x[best_idx].numpy()