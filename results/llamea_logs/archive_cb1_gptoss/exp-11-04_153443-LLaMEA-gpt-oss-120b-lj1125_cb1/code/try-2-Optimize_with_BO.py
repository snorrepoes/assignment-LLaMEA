import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch import settings
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize


class Optimize_with_BO:
    """
    Bayesian Optimizer using BoTorch/GPyTorch.
    Parameters
    ----------
    budget : int
        Maximum number of allowed function evaluations (including the initial DoE).
    dim : int
        Dimensionality of the search space (fixed to 6 for this problem).
    initial_points : np.ndarray, shape (n_init, dim)
        Initial Design of Experiments points; they are evaluated before BO starts.
    random_seed : int
        Seed for all stochastic components (numpy, torch, etc.).
    """

    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = int(budget)
        self.dim = int(dim)
        if initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        self.random_seed = int(random_seed)

        # set all random seeds
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        # for reproducibility on GPU (if ever used)
        torch.cuda.manual_seed_all(self.random_seed)

        # internal counters / storage
        self._n_evals = 0
        self._best_f = np.inf
        self._best_x = None

    def __call__(self, func):
        """
        Run the optimizer on a user supplied function.
        The function must accept a 1‑D numpy array of length ``dim`` and return a float.
        It must also provide ``func.bounds.lb`` and ``func.bounds.ub`` (array‑like of length ``dim``).
        Returns
        -------
        best_f : float
            Best objective value observed.
        best_x : np.ndarray
            Location of the best value.
        """
        # ------------------------------------------------------------------
        # Helper to check budget and raise if exceeded
        def _check_budget():
            if self._n_evals > self.budget:
                raise RuntimeError(
                    f"Evaluation budget exceeded: used {self._n_evals} > {self.budget}"
                )

        # ------------------------------------------------------------------
        # Evaluate initial design
        n_init = self.initial_points.shape[0]
        if n_init > self.budget:
            raise ValueError(
                f"Number of initial points ({n_init}) exceeds the total budget ({self.budget})."
            )
        X_train = []
        Y_train = []

        for i in range(n_init):
            x = self.initial_points[i]
            f_val = func(x)
            X_train.append(x)
            Y_train.append(f_val)
            self._n_evals += 1
            _check_budget()
            if f_val < self._best_f:
                self._best_f = f_val
                self._best_x = x.copy()

        X_train = torch.tensor(np.array(X_train), dtype=torch.double)
        Y_train = torch.tensor(np.array(Y_train).reshape(-1, 1), dtype=torch.double)

        # ------------------------------------------------------------------
        # Bounds for the acquisition optimizer
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            raise ValueError("func.bounds.lb/ub must be array‑like of length `dim`")
        bounds = torch.tensor(np.vstack([lb, ub]), dtype=torch.double)  # 2 x dim

        # ------------------------------------------------------------------
        # Main BO loop
        while self._n_evals < self.budget:
            # Fit GP model
            gp = SingleTaskGP(
                X_train, Y_train, outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # Acquisition function: log Expected Improvement (better for highly skewed targets)
            # We pass the current best (on the transformed space) automatically via model.posterior
            ei = qLogExpectedImprovement(
                model=gp,
                best_f=Y_train.max(),  # best on the *standardized* scale
                sampler=None,  # default Sobol/QMC sampler is fine
            )

            # Optimize acquisition function (single candidate per iteration)
            candidate, acq_value = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"maxiter": 200},
                sequential=True,
            )
            # candidate is a tensor of shape (1, dim)
            x_new = candidate.detach().cpu().numpy().reshape(-1)
            # evaluate the true objective
            f_new = func(x_new)

            # Update counters and best solution
            self._n_evals += 1
            _check_budget()
            if f_new < self._best_f:
                self._best_f = f_new
                self._best_x = x_new.copy()

            # Append new data to training set
            X_train = torch.cat([X_train, torch.tensor([x_new], dtype=torch.double)], dim=0)
            Y_train = torch.cat([Y_train, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

        # ------------------------------------------------------------------
        return float(self._best_f), np.copy(self._best_x)