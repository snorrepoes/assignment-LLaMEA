import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, LogStandardize
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
        self.bounds = torch.tensor([[-5.0]*dim, [5.0]*dim], dtype=torch.double)

    def __call__(self, func):
        X_data = torch.tensor(self.initial_points, dtype=torch.double)
        Y_data = []
        for x in X_data:
            Y_data.append(func(x.numpy()))
        Y_data = torch.tensor(Y_data, dtype=torch.double).unsqueeze(-1)
        
        best_f = Y_data.min().item()
        best_x = X_data[Y_data.argmin()].numpy()

        for _ in range(self.budget - len(X_data)):
            # Log transform Y locally to handle the dynamic range before passing to GP
            # Using OutcomeTransform LogStandardize to ensure numerical stability
            model = SingleTaskGP(
                X_data, Y_data, 
                outcome_transform=LogStandardize(),
                likelihood=GaussianLikelihood(noise_constraint=torch.distributions.constraints.greater_than(1e-6))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Use qLogExpectedImprovement to avoid vanishing gradient in plateaus
            acq_func = qLogExpectedImprovement(model=model, best_f=Y_data.min())
            
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=256
            )
            
            new_x = candidates.detach()
            new_y = func(new_x.squeeze().numpy())
            
            X_data = torch.cat([X_data, new_x])
            Y_data = torch.cat([Y_data, torch.tensor([[new_y]], dtype=torch.double)])
            
            if new_y < best_f:
                best_f = new_y
                best_x = new_x.squeeze().numpy()
                
        return best_f, best_x