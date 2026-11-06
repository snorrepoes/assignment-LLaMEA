import numpy as np
import torch
import warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.optim import optimize_acqf
from scipy.stats import qmc


class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = np.array(initial_points, dtype=np.float64)
        self.random_seed = random_seed
        self.n_calls = 0
        self.best_f = np.inf
        self.best_x = None
        self.X = []
        self.y = []

    def _evaluate(self, func, x):
        self.n_calls += 1
        if self.n_calls > self.budget:
            raise Exception(f"Budget exceeded: call {self.n_calls} > budget {self.budget}")
        f = func(np.array(x, dtype=np.float64))
        self.X.append(np.array(x, dtype=np.float64))
        self.y.append(f)
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=np.float64).copy()
        return f

    def _fit_gp(self):
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float)
        y_tensor = torch.tensor(np.array(self.y), dtype=torch.float).unsqueeze(-1)

        model = SingleTaskGP(
            X=X_tensor,
            y=y_tensor,
            input_transform=Normalize(d=self.dim),
            outcome_transform=Standardize(m=1),
        )
        # Short lengthscale in normalized [0,1]^d space to resolve multimodal peaks
        model.covar_module.base_kernel.lengthscale = torch.full((self.dim,), 0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(model, progress_bar=False)

        return model

    def __call__(self, func):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        dim = self.dim
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        bounds = [torch.tensor(lb, dtype=torch.float), torch.tensor(ub, dtype=torch.float)]

        # Phase 1: Evaluate initial points
        for x in self.initial_points:
            self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # Phase 2: Sobol points to improve initial space coverage
        n_sobol = min(5, self.budget - self.n_calls)
        if n_sobol > 0:
            sampler = qmc.Sobol(d=dim, seed=self.random_seed + 42)
            sobol_samples = sampler.random(n_sobol)
            for i in range(n_sobol):
                if self.n_calls >= self.budget:
                    break
                x = lb + sobol_samples[i] * (ub - lb)
                self._evaluate(func, x)

        if self.n_calls >= self.budget:
            return self.best_f, self.best_x

        # Phase 3: Bayesian Optimization loop
        patience = 0
        max_patience = 8
        total_bo = self.budget - self.n_calls
        bo_step = 0
        best_f_history = self.best_f

        while self.n_calls < self.budget:
            bo_step += 1
            n_points = len(self.X)

            # Patience-based random exploration
            if patience >= max_patience:
                x_new = np.random.uniform(lb, ub)
                self._evaluate(func, x_new)
                if self.best_f < best_f_history - 1e-8:
                    best_f_history = self.best_f
                    patience = 0
                continue

            # Fit GP surrogate
            try:
                model = self._fit_gp()
            except Exception:
                x_new = np.random.uniform(lb, ub)
                self._evaluate(func, x_new)
                if self.best_f < best_f_history - 1e-8:
                    best_f_history = self.best_f
                    patience = 0
                else:
                    patience += 1
                continue

            best_f_tensor = torch.tensor([self.best_f], dtype=torch.float)
            progress = bo_step / max(total_bo, 1)

            # Adaptive acquisition: explore early, exploit later
            if progress < 0.4:
                acq_f = qUpperConfidenceBound(
                    model=model, beta=4.0, q=1
                )
            elif progress < 0.75:
                acq_f = qUpperConfidenceBound(
                    model=model, beta=1.5, q=1
                )
            else:
                acq_f = qLogExpectedImprovement(
                    model=model, best_f=best_f_tensor, q=1
                )

            # Multi-restart acquisition optimization
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = optimize_acqf(
                        acq_function=acq_f,
                        bounds=bounds,
                        q=1,
                        num_restarts=30,
                        raw_samples=512,
                    )
            except Exception:
                x_new = np.random.uniform(lb, ub)
                self._evaluate(func, x_new)
                if self.best_f < best_f_history - 1e-8:
                    best_f_history = self.best_f
                    patience = 0
                else:
                    patience += 1
                continue

            x_new = candidates.numpy().flatten()
            old_best = self.best_f
            self._evaluate(func, x_new)

            # Update patience counter
            if self.best_f < old_best - 1e-8:
                patience = 0
                best_f_history = self.best_f
            else:
                patience += 1

        return self.best_f, self.best_x