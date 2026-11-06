import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.bounds = torch.tensor([[-5.0] * dim, [5.0] * dim], dtype=torch.double)

    def __call__(self, func):
        X_data = torch.tensor(self.initial_points, dtype=torch.double)
        Y_data = []
        for x in X_data:
            Y_data.append(func(x.numpy()))
        
        Y_data = torch.tensor(Y_data, dtype=torch.double).view(-1, 1)
        best_f = Y_data.min().item()
        best_x = X_data[Y_data.argmin()].numpy()
        
        num_evals = len(Y_data)
        while num_evals < self.budget:
            model = SingleTaskGP(
                X_data, Y_data, 
                input_transform=Normalize(d=self.dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            acq_func = qNoisyExpectedImprovement(model=model, X_baseline=X_data)
            
            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512
            )
            
            new_y = func(new_x.detach().squeeze().numpy())
            X_data = torch.cat([X_data, new_x.detach()])
            Y_data = torch.cat([Y_data, torch.tensor([[new_y]], dtype=torch.double)])
            num_evals += 1
            
            if new_y < best_f:
                best_f = new_y
                best_x = new_x.detach().squeeze().numpy()
                
        return float(best_f), best_x