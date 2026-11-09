import numpy as np
import torch
from torch import Tensor
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    """
    Bayesian optimizer using BoTorch.

    Parameters
    ----------
    budget : int
        Total number of allowed evaluations (including the initial DoE points).
    dim : int
        Dimensionality of the search space (here fixed to 6).
    initial_points : np.ndarray, shape (n_init, dim)
        Initial design of experiments (DoE) points, must lie in [-5, 5]^dim.
    random_seed : int
        Seed used for every stochastic component (numpy, torch, Sobol, etc.).
    """

    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        # store arguments
        self.budget = int(budget)
        self.dim = int(dim)
        self.random_seed = int(random_seed)

        # set all random seeds
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        # validate and store initial points
        if initial_points.ndim != 2 or initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must be a 2‑D array with shape (n_init, dim).")
        if not np.all((initial_points >= -5) & (initial_points <= 5)):
            raise ValueError("initial_points must lie inside the box [-5,5]^dim.")
        self.initial_points = initial_points.astype(np.float64)

    def __call__(self, func):
        """
        Run the optimization.

        Parameters
        ----------
        func : callable
            Black‑box objective. Takes a 1‑D numpy array of length ``dim`` and returns a float.
            Must have attributes ``func.bounds.lb`` and ``func.bounds.ub`` (both array‑like of length ``dim``).

        Returns
        -------
        best_f, best_x : float, np.ndarray
            The best function value observed and the corresponding location.
        """
        # ---------------------------- initial evaluations ----------------------------
        n_init = self.initial_points.shape[0]
        if n_init > self.budget:
            raise RuntimeError(
                f"Number of initial points ({n_init}) exceeds the total budget ({self.budget})."
            )

        # storage for observations
        X_obs = []  # list of torch tensors, shape (n, dim)
        Y_obs = []  # list of torch tensors, shape (n, 1)

        best_f = np.inf
        best_x = None

        for i in range(n_init):
            x_np = self.initial_points[i]
            f_val = func(x_np)  # may raise if function fails
            if f_val < best_f:
                best_f = f_val
                best_x = x_np.copy()
            # store as torch tensors (float64)
            X_obs.append(torch.from_numpy(x_np).double())
            Y_obs.append(torch.tensor([f_val], dtype=torch.double))

        # convert lists to tensors
        X_train = torch.stack(X_obs)                # (n_init, dim)
        Y_train = torch.stack(Y_obs)                # (n_init, 1)

        # remaining budget
        remaining = self.budget - n_init

        # bounds for acquisition optimisation (torch tensors)
        bounds = torch.tensor(
            np.vstack([func.bounds.lb, func.bounds.ub]), dtype=torch.double
        )  # shape (2, dim)

        # ---------------------------- BO loop ----------------------------
        while remaining > 0:
            # ----- 1. Fit a GP surrogate -----
            # Normalise outputs (standardization) to aid optimisation
            gp = SingleTaskGP(
                X_train,
                Y_train,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # ----- 2. Acquisition -----
            # Use Expected Improvement (EI) with the current best_f
            ei = ExpectedImprovement(
                model=gp,
                best_f=best_f,
                maximize=False,  # we minimise the black‑box function
            )

            # ----- 3. Optimise acquisition to propose next point -----
            # Use a modest number of restarts / raw samples for speed
            # (still deterministic because Sobol draws are seeded through torch's seed)
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"disp": False},
            )
            # candidate is shape (1, dim)
            x_next = candidate.squeeze(0)  # (dim,)

            # ----- 4. Evaluate objective -----
            x_np = x_next.detach().cpu().numpy()
            f_val = func(x_np)

            # ----- 5. Update storage and best solution -----
            X_train = torch.cat([X_train, x_next.unsqueeze(0)], dim=0)
            Y_train = torch.cat([Y_train, torch.tensor([[f_val]], dtype=torch.double)], dim=0)

            if f_val < best_f:
                best_f = f_val
                best_x = x_np.copy()

            remaining -= 1

        # Final result
        return float(best_f), np.array(best_x, copy=False)