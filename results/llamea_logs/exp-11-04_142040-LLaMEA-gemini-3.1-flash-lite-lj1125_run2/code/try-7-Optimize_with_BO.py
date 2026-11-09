import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.seed = random_seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def __call__(self, func):
        bounds = torch.stack([torch.tensor(func.bounds.lb), torch.tensor(func.bounds.ub)])
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        Y_obs = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        if len(X_obs) > self.budget:
            raise ValueError("Initial points exceed budget.")

        while len(X_obs) < self.budget:
            # Use Log-transformation on objective to handle deep narrow wells and plateaus
            # Minimize -log(y - min + eps) is equivalent to maximizing log(y - min + eps)
            y_min = Y_obs.min()
            train_Y = -torch.log(Y_obs - y_min + 1e-9)
            
            # ARD Matern kernel for better handling of different dimensional scales
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(
                train_X=normalize(X_obs, bounds=bounds),
                train_Y=train_Y,
                covar_module=covar_module,
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Maximize EI on the transformed space (Log-EI)
            acq = LogExpectedImprovement(
                model=model,
                best_f=train_Y.max(),
                maximize=True
            )

            new_x_norm, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                q=1,
                num_restarts=20,
                raw_samples=1024
            )
            
            new_x = unnormalize(new_x_norm, bounds=bounds)
            new_f = func(new_x.detach().numpy().flatten())
            
            X_obs = torch.cat([X_obs, new_x])
            Y_obs = torch.cat([Y_obs, torch.tensor([[new_f]], dtype=torch.double)])

        best_idx = torch.argmin(Y_obs)
        return Y_obs[best_idx].item(), X_obs[best_idx].numpy()