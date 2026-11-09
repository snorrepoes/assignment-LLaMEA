import numpy as np
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch import settings
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.sampling import SobolQMCNormalSampler

class Optimize_with_BO:
    """
    Bayesian optimisation wrapper that respects a hard evaluation budget.

    Parameters
    ----------
    budget : int
        Maximum number of calls to ``func`` (including the initial DoE).
    dim : int
        Dimensionality of the search space (must match ``initial_points``).
    initial_points : np.ndarray
        Array of shape (n_init, dim) containing the initial design of experiments.
        Points are evaluated *before* any BO iteration.
    random_seed : int
        Seed used for all stochastic components (numpy, torch, Sobol, etc.).
    """

    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = int(budget)
        self.dim = int(dim)
        if initial_points.shape[1] != dim:
            raise ValueError("initial_points must have shape (n_init, dim)")
        self.initial_X = np.asarray(initial_points, dtype=np.float64)
        self.n_init = self.initial_X.shape[0]

        # store seed and make reproducible
        self.seed = int(random_seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # placeholders for the best found solution
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func) -> tuple[float, np.ndarray]:
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            A function that accepts a 1‑D ``np.ndarray`` of length ``dim`` and returns a float.
            It must expose ``func.bounds.lb`` and ``func.bounds.ub`` (array‑like).

        Returns
        -------
        best_f, best_x : tuple
            Minimum objective value observed and the corresponding point.
        """
        # ------------------------------------------------------------------
        # Helper to count calls and raise if budget would be exceeded
        # ------------------------------------------------------------------
        calls = 0
        def _check_budget():
            nonlocal calls
            if calls >= self.budget:
                raise RuntimeError(
                    f"Evaluation budget of {self.budget} exceeded. "
                    f"Already performed {calls} calls."
                )
        # ------------------------------------------------------------------
        # 1) Evaluate the supplied initial design (DoE)
        # ------------------------------------------------------------------
        X_list = []
        Y_list = []

        for x_np in self.initial_X:
            _check_budget()
            f_val = float(func(x_np))
            calls += 1
            X_list.append(x_np)
            Y_list.append([f_val])  # keep 2‑D for GP
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_np.copy()

        # Convert to tensors (double precision for stability)
        train_X = torch.tensor(np.vstack(X_list), dtype=torch.double)  # (N, dim)
        train_Y = torch.tensor(np.vstack(Y_list), dtype=torch.double)  # (N, 1)

        # ------------------------------------------------------------------
        # 2) Main BO loop – keep evaluating until the budget is exhausted
        # ------------------------------------------------------------------
        while calls < self.budget:
            # ---- Fit GP ----------------------------------------------------
            # Standardise the output (zero mean, unit var) – essential when values span many orders.
            gp = SingleTaskGP(
                train_X,
                train_Y,
                outcome_transform=Standardize(m=1),
                covar_module=torch.nn.Module(),
            )
            gp.covar_module = torch.nn.Sequential(
                gp.covar_module,
                # Matern kernel with ARD (different lengthscale per dimension)
                gp.covar_module.__class__(nu=2.5, ard_num_dims=self.dim)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # ---- Acquisition (log‑EI is robust to huge dynamic range) -----
            # The acquisition is maximised, we therefore minimise -acq.
            acquisition = LogExpectedImprovement(
                model=gp,
                best_f=train_Y.max(),   # log‑EI uses the **log** of the objective
                maximize=False,
                sampler=SobolQMCNormalSampler(num_samples=128),
            )

            # ---- Optimise acquisition function ---------------------------
            # Use a modest number of restarts (5) and 20 raw samples – cheap given the tiny dim.
            bounds = torch.stack(
                [torch.tensor(func.bounds.lb, dtype=torch.double),
                 torch.tensor(func.bounds.ub, dtype=torch.double)]
            )
            # Ensure bounded in the prescribed domain ([-5,5] for each dim)
            # The optimise_acqf routine works with double tensors.
            candidate, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
                options={"maxiter": 200},
            )
            x_new = candidate.detach().cpu().numpy().reshape(-1)

            # ---- Evaluate the new point ------------------------------------
            _check_budget()
            f_new = float(func(x_new))
            calls += 1

            # ---- Update training data --------------------------------------
            train_X = torch.cat([train_X, torch.tensor([x_new], dtype=torch.double)], dim=0)
            train_Y = torch.cat([train_Y, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

            # ---- Keep track of the best observed value --------------------
            if f_new < self.best_f:
                self.best_f = f_new
                self.best_x = x_new.copy()

        # ------------------------------------------------------------------
        # All budget used – return best result
        # ------------------------------------------------------------------
        return float(self.best_f), np.asarray(self.best_x, dtype=np.float64)