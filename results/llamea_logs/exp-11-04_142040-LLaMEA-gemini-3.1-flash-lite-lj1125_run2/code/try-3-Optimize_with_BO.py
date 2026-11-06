import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
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
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        Y_obs = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        if len(X_obs) > self.budget:
            raise ValueError("Initial points exceed budget.")

        while len(X_obs) < self.budget:
            # Transform objective to mitigate plateau/scale issues
            # We use a log transformation if values are positive, else just standardizing
            train_Y = torch.log(Y_obs - Y_obs.min() + 1e-9)
            
            model = SingleTaskGP(
                train_X=normalize(X_obs, bounds=torch.stack([lb, ub])),
                train_Y=train_Y,
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=normalize(X_obs, bounds=torch.stack([lb, ub])),
                prune_baseline=True
            )

            new_x_norm, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                q=1,
                num_restarts=10,
                raw_samples=512
            )
            
            new_x = unnormalize(new_x_norm, bounds=torch.stack([lb, ub]))
            new_f = func(new_x.detach().numpy().flatten())
            
            X_obs = torch.cat([X_obs, new_x])
            Y_obs = torch.cat([Y_obs, torch.tensor([[new_f]], dtype=torch.double)])

        best_idx = torch.argmin(Y_obs)
        return Y_obs[best_idx].item(), X_obs[best_idx].numpy()