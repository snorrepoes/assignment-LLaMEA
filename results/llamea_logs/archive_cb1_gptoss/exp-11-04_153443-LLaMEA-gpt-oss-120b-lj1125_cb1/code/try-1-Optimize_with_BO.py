import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    """
    Bayesian Optimizer that respects a fixed evaluation budget.
    It evaluates the supplied initial design first and then runs a
    standard BO loop with a GP surrogate and Expected Improvement.
    """

    def __init__(self, budget: int = 60, dim: int = 6,
                 initial_points: np.ndarray = None, random_seed: int = 0):
        """
        Parameters
        ----------
        budget: int
            Total number of allowed function evaluations (including initial points).
        dim: int
            Dimension of the search space.
        initial_points: np.ndarray, shape (n_init, dim)
            Initial design of experiments. Must lie in the bounds [-5, 5]^dim.
        random_seed: int
            Seed for all stochastic components (numpy, torch, Sobol/LHS if used).
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.random_seed = int(random_seed)

        # seed everything
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        if initial_points is None:
            raise ValueError("initial_points must be provided")
        if initial_points.shape[1] != self.dim:
            raise ValueError(f"initial_points must have shape (n_init, {self.dim})")
        if not np.all((-5 <= initial_points) & (initial_points <= 5)):
            raise ValueError("initial_points must lie inside [-5, 5]^dim")

        self.initial_points = initial_points.astype(np.float64)

        # bounds (torch tensor of shape 2 x dim)
        self.bounds = torch.tensor(
            [np.full(self.dim, -5.0), np.full(self.dim, 5.0)],
            dtype=torch.double,
        )

    def __call__(self, func):
        """
        Runs the BO procedure on the provided black‑box function.

        Parameters
        ----------
        func: Callable
            A function that receives a 1‑D numpy array of length `dim`
            and returns a scalar float. It must also provide `func.bounds.lb`
            and `func.bounds.ub` (arrays of length `dim`) which are ignored
            here because the optimizer uses the fixed [-5,5]^dim bounds.

        Returns
        -------
        best_f: float
            Lowest observed function value.
        best_x: np.ndarray, shape (dim,)
            Corresponding location.
        """
        # ------------------------------------------------------------------
        # 1) Evaluate the initial design (counts against the budget)
        # ------------------------------------------------------------------
        n_init = self.initial_points.shape[0]
        if n_init > self.budget:
            raise RuntimeError(
                f"Number of initial points ({n_init}) exceeds the total budget ({self.budget})."
            )

        X = torch.tensor(self.initial_points, dtype=torch.double)  # (n_init, dim)
        Y_vals = []
        for xi in self.initial_points:
            fval = float(func(xi))
            Y_vals.append(fval)
        Y = torch.tensor(Y_vals, dtype=torch.double).unsqueeze(-1)  # (n_init, 1)

        best_f = float(Y.min())
        best_x = X[Y.argmin()].cpu().numpy().copy()
        n_evals = n_init

        # ------------------------------------------------------------------
        # 2) Main BO loop
        # ------------------------------------------------------------------
        while n_evals < self.budget:
            # ---- Fit GP ----------------------------------------------------
            gp = SingleTaskGP(
                X, Y,
                outcome_transform=Standardize(m=1)  # standardize the outputs
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # ---- Acquisition (EI) -----------------------------------------
            ei = ExpectedImprovement(
                model=gp,
                best_f=best_f,
                maximize=False,  # we minimize the objective
            )

            # ---- Optimize acquisition ---------------------------------------
            # q=1, batch size 1 (each iteration consumes one evaluation)
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 200},
            )  # candidate shape (1, dim)

            # ---- Evaluate new point ----------------------------------------
            new_x = candidate.squeeze(0).detach().cpu().numpy()
            f_new = float(func(new_x))

            # ---- bookkeeping ------------------------------------------------
            n_evals += 1
            if f_new < best_f:
                best_f = f_new
                best_x = new_x.copy()

            # Append to training data
            X = torch.cat([X, candidate], dim=0)  # (n+1, dim)
            Y = torch.cat([Y, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

        # ------------------------------------------------------------------
        # 3) Return the best result found
        # ------------------------------------------------------------------
        return best_f, best_x