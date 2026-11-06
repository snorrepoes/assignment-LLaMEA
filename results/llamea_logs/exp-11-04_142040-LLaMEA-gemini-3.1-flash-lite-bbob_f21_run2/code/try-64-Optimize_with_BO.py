import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.normal import SobolQMCNormalSampler

class Optimize_with_BO:
    def __init__(self, budget=60, dim=6, initial_points=None, random_seed=42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        bounds = torch.tensor([func.bounds.lb, func.bounds.ub], dtype=torch.double)
        train_x = torch.tensor(self.initial_points, dtype=torch.double)
        train_y = torch.tensor([func(x) for x in train_x], dtype=torch.double).view(-1, 1)
        
        num_calls = len(train_x)
        length = 0.5
        success_counter = 0
        failure_counter = 0
        
        while num_calls < self.budget:
            train_x_norm = normalize(train_x, bounds)
            
            likelihood = GaussianLikelihood()
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
            model = SingleTaskGP(train_x_norm, train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq_func = qNoisyExpectedImprovement(
                model=model,
                X_baseline=train_x_norm,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            )

            best_idx = torch.argmin(train_y)
            center = train_x_norm[best_idx]
            tr_lb = torch.clamp(center - length, 0.0, 1.0)
            tr_ub = torch.clamp(center + length, 0.0, 1.0)

            candidate_norm, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1, num_restarts=10, raw_samples=256,
            )

            new_x = unnormalize(candidate_norm, bounds).detach().squeeze(0).numpy()
            new_y = func(new_x)
            
            train_x = torch.cat([train_x, torch.tensor([new_x], dtype=torch.double)])
            train_y = torch.cat([train_y, torch.tensor([[new_y]], dtype=torch.double)])
            
            if new_y < train_y[best_idx]:
                success_counter += 1
                failure_counter = 0
                if success_counter >= 3:
                    length = min(length * 2.0, 1.0)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= 2:
                    length /= 2.0
                    failure_counter = 0
            
            if length < 1e-3: break
            num_calls += 1

        best_idx = torch.argmin(train_y)
        return train_y[best_idx].item(), train_x[best_idx].numpy()