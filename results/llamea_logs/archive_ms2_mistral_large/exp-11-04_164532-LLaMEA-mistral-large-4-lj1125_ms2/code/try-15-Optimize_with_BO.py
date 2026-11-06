import warnings
import random
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=float)
        if self.initial_points.ndim != 2 or self.initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        self.random_seed = int(random_seed)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self._call_count = 0
        self._X = []
        self._Y = []
        self._best_f = np.inf
        self._best_x = None

    def _evaluate(self, func, x):
        if self._call_count >= self.budget:
            raise RuntimeError("Function call budget exceeded")
        x = np.asarray(x, dtype=float).reshape(self.dim)
        y = float(func(x))
        self._call_count += 1
        self._X.append(x)
        self._Y.append(y)
        if y < self._best_f:
            self._best_f = y
            self._best_x = x.copy()
        return y

    def __call__(self, func):
        self._call_count = 0
        self._X = []
        self._Y = []
        self._best_f = np.inf
        self._best_x = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        lb = np.asarray(func.bounds.lb, dtype=float).reshape(self.dim)
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(self.dim)

        n_init = self.initial_points.shape[0]
        if n_init > self.budget:
            raise ValueError("Number of initial points exceeds the evaluation budget")

        for x in self.initial_points:
            self._evaluate(func, x)

        if n_init == 0:
            n_rand = min(5, self.budget)
            for _ in range(n_rand):
                x = np.random.uniform(lb, ub)
                self._evaluate(func, x)

        while self._call_count < self.budget:
            X = np.asarray(self._X, dtype=float)
            Y = np.asarray(self._Y, dtype=float)

            # Sign-log1p compression followed by standardization
            Y_trans = np.sign(Y) * np.log1p(np.abs(Y))
            mean = float(np.mean(Y_trans))
            std = float(np.std(Y_trans))
            if std < 1e-12:
                std = 1.0
            Y_std = (Y_trans - mean) / std

            train_X = torch.tensor(X, dtype=torch.double)
            train_Y = torch.tensor(Y_std, dtype=torch.double).unsqueeze(-1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SingleTaskGP(train_X, train_Y, outcome_transform=None)
                model.train()

                best_trans = float(np.min(Y_std))
                acq = qLogExpectedImprovement(model=model, best_f=best_trans)

                bounds = torch.tensor(np.vstack([lb, ub]), dtype=torch.double)
                try:
                    candidate, _ = optimize_acqf(
                        acq,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                    )
                    x_next = candidate.detach().cpu().numpy().reshape(-1)
                    if not np.all(np.isfinite(x_next)):
                        raise ValueError("Invalid acquisition candidate")
                except Exception:
                    x_next = np.random.uniform(lb, ub)

            x_next = np.clip(x_next, lb, ub)
            self._evaluate(func, x_next)

        return self._best_f, self._best_x