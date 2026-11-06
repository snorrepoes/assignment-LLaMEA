import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def _rank_transform(self, y):
        """Map observed values to [0,1] via rank: best→0, worst→1.
        This amplifies tiny differences in a flat plateau and compresses
        large-scale variation, making the GP target well-conditioned."""
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        if n <= 1:
            return torch.tensor([0.0], dtype=torch.float32)
        ranks = np.argsort(np.argsort(y))  # 0 = best, n-1 = worst
        return torch.tensor(ranks / (n - 1), dtype=torch.float32)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lb = torch.tensor(np.asarray(func.bounds.lb, dtype=np.float32))
        ub = torch.tensor(np.asarray(func.bounds.ub, dtype=np.float32))

        X_list = []
        f_list = []
        total_calls = 0

        # --- Evaluate initial DoE (counts against budget) ---
        for x in self.initial_points:
            if total_calls >= budget:
                raise RuntimeError("Budget exhausted during initial evaluations.")
            f = float(func(x))
            X_list.append(np.asarray(x, dtype=np.float64))
            f_list.append(f)
            total_calls += 1

        f_best = min(f_list)
        x_best = X_list[int(np.argmin(f_list))].copy()

        # --- Bayesian Optimization loop ---
        n_init = len(self.initial_points)
        while total_calls < budget:
            n_obs = len(X_list)
            x_new = None

            # Use GP+UCB only when we have enough distinct observations
            if n_obs >= self.dim + 2 and np.ptp(f_list) > 1e-14:
                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                f_ranked = self._rank_transform(f_list).unsqueeze(-1)

                try:
                    model = SingleTaskGP(X, f_ranked)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll)
                    model.eval()

                    # Adaptive beta: more exploration early, exploitation later
                    bo_iter = total_calls - n_init
                    n_bo_total = budget - n_init
                    progress = bo_iter / max(n_bo_total, 1)
                    beta = 8.0 * (1.0 - progress) ** 0.5 + 0.5

                    with torch.no_grad():
                        acqf = qUpperConfidenceBound(model=model, beta=beta)
                        candidates, _ = optimize_acqf(
                            acq_function=acqf,
                            bounds=(lb, ub),
                            q=1,
                            num_restarts=30,
                            raw_samples=4096,
                            options={"maxiter": 200},
                        )
                    x_new = candidates.flatten().numpy()

                except Exception:
                    x_new = None

            # Fallback: uniform random
            if x_new is None:
                x_new = np.random.uniform(lb.numpy(), ub.numpy())

            f_val = float(func(x_new))
            X_list.append(x_new)
            f_list.append(f_val)
            total_calls += 1

            if f_val < f_best:
                f_best = f_val
                x_best = x_new.copy()

        return f_best, x_best