import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub])
        
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in self.initial_points], dtype=torch.double).unsqueeze(-1)
        
        num_calls = len(train_x)
        length = 1.6
        min_length = 0.05
        max_length = 1.6
        success_counter = 0
        failure_counter = 0

        while num_calls < self.budget:
            train_y_min = train_y.min()
            
            model = SingleTaskGP(
                train_x, train_y, 
                input_transform=Normalize(d=self.dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_x = train_x[train_y.argmin()]
            tr_lb = torch.clamp(best_x - length, lb, ub)
            tr_ub = torch.clamp(best_x + length, lb, ub)
            tr_bounds = torch.stack([tr_lb, tr_ub])

            # Use qNEI for exploitation within the trust region
            acq = qNoisyExpectedImprovement(model, X_baseline=train_x, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])))
            
            candidate, _ = optimize_acqf(
                acq_function=acq, bounds=tr_bounds, q=1, num_restarts=10, raw_samples=512
            )

            new_x = candidate.detach().flatten().numpy()
            new_y = func(new_x)
            
            if new_y < train_y_min:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    length = min(2.0 * length, max_length)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= 3:
                    length = max(length / 2.0, min_length)
                    failure_counter = 0
            
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()