import numpy as np
import torch
from torch import tensor
import random

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    """
    Bayesian optimisation for a 6‑dimensional black‑box function.
    Parameters
    ----------
    budget : int
        Maximum number of calls to ``func`` (including the initial DoE).
    dim : int
        Dimensionality of the search space (expected 6).
    initial_points : np.ndarray, shape (n_init, dim)
        Design‑of‑experiments points that are evaluated first.
    random_seed : int
        Seed used for every stochastic component (numpy, torch, random).
    """

    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = int(budget)
        self.dim = int(dim)
        if initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        self.random_seed = int(random_seed)

        # deterministic behaviour
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

        # internal bookkeeping
        self._n_evals = 0          # number of evaluations performed
        self._X = None             # torch tensor, shape (n, dim)
        self._Y = None             # torch tensor, shape (n, 1)

    def __call__(self, func):
        """
        Run the optimisation and return best value and location.
        Parameters
        ----------
        func : callable
            Function to minimise.  ``func`` receives a 1‑D ``np.ndarray`` of length ``dim``
            and returns a scalar ``float``.  It must expose ``func.bounds.lb`` and
            ``func.bounds.ub`` as 1‑D ``np.ndarray`` objects.
        Returns
        -------
        best_f : float
            Minimum objective value observed.
        best_x : np.ndarray, shape (dim,)
            Location of the minimum.
        """
        # ------------------------------------------------------------------ #
        # Helper to check budget and raise if we would exceed it
        # ------------------------------------------------------------------ #
        def _budget_check():
            if self._n_evals >= self.budget:
                raise RuntimeError(
                    f"Evaluation budget of {self.budget} exceeded (already used {self._n_evals})."
                )

        # ------------------------------------------------------------------ #
        # Prepare bounds
        # ------------------------------------------------------------------ #
        bounds_np = np.column_stack((func.bounds.lb, func.bounds.ub))  # (dim, 2)
        if bounds_np.shape != (self.dim, 2):
            raise ValueError("Bounds shape mismatch with problem dimension.")
        bounds = torch.tensor(bounds_np, dtype=torch.double).t()  # (2, dim) for BoTorch

        # ------------------------------------------------------------------ #
        # Evaluate initial DoE points
        # ------------------------------------------------------------------ #
        X_list = []
        Y_list = []
        best_f = np.inf
        best_x = None

        for x_np in self.initial_points:
            _budget_check()
            f_val = float(func(x_np))
            self._n_evals += 1

            X_list.append(x_np)
            Y_list.append([f_val])

            if f_val < best_f:
                best_f = f_val
                best_x = x_np.copy()

        # store as torch tensors (double precision)
        self._X = torch.tensor(np.asarray(X_list), dtype=torch.double)
        self._Y = torch.tensor(np.asarray(Y_list), dtype=torch.double)

        # ------------------------------------------------------------------ #
        # Main BO loop
        # ------------------------------------------------------------------ #
        while self._n_evals < self.budget:
            # ---- Fit GP --------------------------------------------------- #
            gp = SingleTaskGP(
                self._X,
                self._Y,
                outcome_transform=Standardize(m=1),   # standardise outputs
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # ---- Acquisition (log‑EI) ------------------------------------ #
            best_f_sofar = self._Y.min()
            acquisition = qLogExpectedImprovement(
                model=gp,
                best_f=best_f_sofar,
                max_log=4,                      # log‑scaling to cope with huge dynamic range
                objective=None,
                sampler=None,
                cache_root=None,
                maximize=False,
            )

            # ---- Optimize acquisition ------------------------------------- #
            # use a fairly large number of raw samples to explore the flat plateau
            candidate, acq_value = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )
            x_new = candidate.detach().cpu().numpy().reshape(-1)  # (dim,)

            # ---- Evaluate ------------------------------------------------- #
            _budget_check()
            f_new = float(func(x_new))
            self._n_evals += 1

            # ---- Update data ----------------------------------------------- #
            self._X = torch.cat([self._X, torch.tensor(x_new, dtype=torch.double).unsqueeze(0)])
            self._Y = torch.cat([self._Y, torch.tensor([[f_new]], dtype=torch.double)])

            # ---- Track best ------------------------------------------------ #
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        return float(best_f), best_x