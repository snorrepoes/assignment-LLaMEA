import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    """
    Bayesian optimisation for a black‑box function with a fixed evaluation budget.

    Parameters
    ----------
    budget : int
        Total number of allowed calls to ``func`` (including the initial points).
    dim : int
        Dimensionality of the search space (must be 6 for the problem at hand).
    initial_points : np.ndarray, shape (n_init, dim)
        Initial DoE points in the domain ``[-5, 5]^dim``.
    random_seed : int
        Seed used for *all* stochastic components (numpy, torch, Sobol, …).
    """

    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        if self.initial_points.ndim != 2 or self.initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        self.random_seed = int(random_seed)

        # set seeds for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self._calls = 0  # internal counter

    def __call__(self, func):
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            Function to minimise.  It must accept a 1‑D NumPy array of length ``dim`` and
            return a scalar float.  ``func.bounds.lb`` and ``func.bounds.ub`` must be
            iterable objects of length ``dim`` that give the lower and upper bounds.

        Returns
        -------
        best_f : float
            Lowest function value observed.
        best_x : np.ndarray, shape (dim,)
            Location where ``best_f`` was observed.
        """
        # --------------------------------------------------------------
        # 1) evaluate the supplied initial design (counts toward budget)
        # --------------------------------------------------------------
        n_init = self.initial_points.shape[0]
        if n_init > self.budget:
            raise RuntimeError(
                f"The number of initial points ({n_init}) exceeds the total budget ({self.budget})."
            )

        X_list = []
        Y_list = []
        for x_np in self.initial_points:
            f_val = float(func(x_np))
            self._calls += 1
            if self._calls > self.budget:
                raise RuntimeError("Budget exceeded while evaluating initial points.")
            X_list.append(x_np.copy())
            Y_list.append(f_val)

        X = np.vstack(X_list)                     # shape (n, dim)
        Y = np.asarray(Y_list, dtype=np.float64)  # shape (n,)

        # keep track of the best observed value / location
        best_idx = int(np.argmin(Y))
        best_f = float(Y[best_idx])
        best_x = X[best_idx].copy()

        # --------------------------------------------------------------
        # 2) Bayesian optimisation loop (single‑point batches)
        # --------------------------------------------------------------
        remaining = self.budget - self._calls
        if remaining == 0:
            return best_f, best_x

        # torch tensors (double precision)
        X_t = torch.from_numpy(X).double()
        Y_t = torch.from_numpy(Y).double().unsqueeze(-1)  # shape (n, 1)

        # box bounds taken from the provided function object
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub])  # shape (2, dim)

        while self._calls < self.budget:
            # ---- fit GP surrogate -------------------------------------------------
            gp = SingleTaskGP(X_t, Y_t)                     # Matern kernel + ARD by default
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # ---- acquisition function (Expected Improvement) --------------------
            ei = ExpectedImprovement(model=gp, best_f=torch.tensor(best_f, dtype=torch.double))

            # ---- optimise acquisition to get next candidate ------------------------
            # we use a few restarts and many raw samples (cheap given the tiny budget)
            cand, _ = optimize_acqf(
                ei,
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )
            x_next = cand.detach().cpu().numpy().reshape(-1)

            # enforce hard bounds (numeric jitter)
            x_next = np.clip(x_next, func.bounds.lb, func.bounds.ub)

            # ---- evaluate the black‑box function ---------------------------------
            f_next = float(func(x_next))
            self._calls += 1
            if self._calls > self.budget:
                raise RuntimeError("Budget exceeded while evaluating BO proposals.")

            # ---- update best --------------------------------------------------------
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()

            # ---- augment dataset ----------------------------------------------------
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            X_t = torch.from_numpy(X).double()
            Y_t = torch.from_numpy(Y).double().unsqueeze(-1)

        return best_f, best_x