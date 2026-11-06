import numpy as np
import torch
from torch import tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

class BudgetExceeded(RuntimeError):
    """Raised when more function evaluations are requested than the allotted budget."""
    pass


class Optimize_with_BO:
    """
    Bayesian optimizer using BoTorch.

    Parameters
    ----------
    budget : int
        Total number of allowed evaluations (including the initial DoE points).
    dim : int
        Dimensionality of the search space (fixed to 6 in the problem statement).
    initial_points : np.ndarray
        Array of shape (n_init, dim) with points inside the bounds; they are evaluated first.
    random_seed : int
        Seed used for all stochastic components (numpy, torch, Sobol, etc.).
    """

    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.X_init = np.asarray(initial_points, dtype=np.float64)
        if self.X_init.ndim != 2 or self.X_init.shape[1] != self.dim:
            raise ValueError("initial_points must be of shape (n_init, dim)")
        self.n_init = self.X_init.shape[0]
        if self.n_init > self.budget:
            raise ValueError("Number of initial points exceeds the total budget")
        self.seed = int(random_seed)

        # set seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __call__(self, func):
        """
        Run the optimizer on the provided black‑box function.

        Parameters
        ----------
        func : callable
            Function to minimize.  Takes a 1‑D numpy array of length `dim` and returns a float.
            Must expose `func.bounds.lb` and `func.bounds.ub` as array‑like objects.

        Returns
        -------
        best_f : float
            Best (lowest) function value observed.
        best_x : np.ndarray
            Corresponding point of shape (dim,).
        """
        # ---------------------------------------------------------------------
        # 1. Extract bounds and convert to torch tensors
        # ---------------------------------------------------------------------
        lb = np.asarray(func.bounds.lb, dtype=np.float64).reshape(1, -1)
        ub = np.asarray(func.bounds.ub, dtype=np.float64).reshape(1, -1)
        if lb.shape[1] != self.dim or ub.shape[1] != self.dim:
            raise ValueError("Function bounds dimension does not match optimizer dimension")

        bounds = torch.tensor(np.vstack([lb, ub]), dtype=torch.double)  # shape (2, dim)

        # ---------------------------------------------------------------------
        # 2. Evaluate the initial design (counts toward the budget)
        # ---------------------------------------------------------------------
        X = torch.tensor(self.X_init, dtype=torch.double, requires_grad=False)
        Y_vals = []
        for x_np in self.X_init:
            y = func(x_np)
            Y_vals.append(y)
        Y = torch.tensor(Y_vals, dtype=torch.double).unsqueeze(-1)  # shape (n_init, 1)

        eval_counter = self.n_init

        # Keep track of the best observed value/location
        best_f, best_idx = torch.min(Y, dim=0)
        best_x = X[best_idx.item()].clone()

        # ---------------------------------------------------------------------
        # 3. Main BO loop (single point batch size)
        # ---------------------------------------------------------------------
        while eval_counter < self.budget:
            # -------------------------------------------------------------
            # 3.1 Fit a GP surrogate to the data gathered so far
            # -------------------------------------------------------------
            gp = SingleTaskGP(X, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # -------------------------------------------------------------
            # 3.2 Define the acquisition function (Expected Improvement)
            # -------------------------------------------------------------
            ei = qExpectedImprovement(
                model=gp,
                best_f=best_f,
                maximize=False,  # we minimise the objective
                batch_size=1,
            )

            # -------------------------------------------------------------
            # 3.3 Optimize the acquisition function to propose the next x
            # -------------------------------------------------------------
            # Use a modest number of restarts and raw samples; they are cheap
            # compared to the expensive black‑box evaluation.
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=256,
                options={"maxiter": 200},
                sequential=True,
            )
            x_next = candidate.detach().numpy().reshape(-1)  # shape (dim,)

            # -------------------------------------------------------------
            # 3.4 Evaluate the black‑box function
            # -------------------------------------------------------------
            if eval_counter + 1 > self.budget:
                raise BudgetExceeded("Attempted to exceed the evaluation budget.")
            y_next = func(x_next)

            # -------------------------------------------------------------
            # 3.5 Append the new observation
            # -------------------------------------------------------------
            X = torch.cat([X, candidate], dim=0)
            Y = torch.cat([Y, torch.tensor([[y_next]], dtype=torch.double)], dim=0)
            eval_counter += 1

            # -------------------------------------------------------------
            # 3.6 Update the incumbent best solution
            # -------------------------------------------------------------
            if y_next < best_f.item():
                best_f = torch.tensor([y_next], dtype=torch.double)
                best_x = torch.tensor(x_next, dtype=torch.double)

        # ---------------------------------------------------------------------
        # 4. Return the result as native Python types
        # ---------------------------------------------------------------------
        return best_f.item(), best_x.numpy()