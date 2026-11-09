import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.generation import MaxPosteriorSampling

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
        bounds = torch.stack([lb, ub])
        
        X_obs = torch.tensor(self.initial_points, dtype=torch.double)
        Y_obs = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        # TuRBO state parameters
        length_min, length_max = 0.5, 2.0
        length = 0.8
        
        while len(X_obs) < self.budget:
            # Model with local standardizing to handle dynamic ranges
            model = SingleTaskGP(
                train_X=normalize(X_obs, bounds),
                train_Y=Y_obs,
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Maximize posterior via Thompson Sampling in the trust region
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            
            # Simplified Trust Region approach: sample near best point
            best_idx = torch.argmin(Y_obs).item()
            best_x = X_obs[best_idx]
            
            # Generate candidates via random sampling in the trust region
            candidates = best_x + length * (torch.rand(512, self.dim, dtype=torch.double) * 2 - 1)
            candidates = torch.clamp(candidates, lb, ub)
            
            with torch.no_grad():
                posterior = model.posterior(normalize(candidates, bounds))
                samples = sampler(candidates.unsqueeze(1))
                new_x = candidates[torch.argmin(samples.squeeze())]
            
            new_f = func(new_x.numpy())
            X_obs = torch.cat([X_obs, new_x.unsqueeze(0)])
            Y_obs = torch.cat([Y_obs, torch.tensor([[new_f]], dtype=torch.double)])
            
            # Simple trust region update logic
            if new_f < Y_obs[best_idx]:
                length = min(length * 1.1, length_max)
            else:
                length = max(length * 0.8, length_min)

        best_idx = torch.argmin(Y_obs)
        return Y_obs[best_idx].item(), X_obs[best_idx].numpy()