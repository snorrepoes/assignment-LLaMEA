import numpy as np
import torch

# BoTorch / GPyTorch imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimize_with_BO:
    """
    Bayesian optimisation for a 6‑dimensional black‑box function.

    Parameters
    ----------
    budget : int
        Total number of allowed calls to ``func`` (including initial DoE points).
    dim : int
        Dimensionality of the search space (must be 6 for this class).
    initial_points : np.ndarray, shape (n_init, dim)
        Design‑of‑experiment points that are evaluated *before* the BO loop.
    random_seed : int
        Seed for all stochastic components (numpy, torch, Sobol, etc.).
    """

    def __init__(self, budget: int, dim: int, initial_points: np.ndarray, random_seed: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.initial_points = np.asarray(initial_points, dtype=np.float64)
        if self.initial_points.ndim != 2 or self.initial_points.shape[1] != self.dim:
            raise ValueError("initial_points must be of shape (n_init, dim)")
        self.random_seed = int(random_seed)

        # deterministic behaviour
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float64)

    def __call__(self, func):
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            A function that receives a 1‑D numpy array of length ``dim`` and returns a float.
            It must expose ``func.bounds.lb`` and ``func.bounds.ub`` (array‑like) defining the
            search hyper‑cube.  The bounds are assumed to be ``[-5, 5]`` in each dimension.

        Returns
        -------
        best_f : float
            Lowest function value observed.
        best_x : np.ndarray, shape (dim,)
            Location where ``best_f`` was obtained.
        """
        # ------------------------------------------------------------------ #
        # 1. extract bounds, ensure they are compatible with the problem.
        # ------------------------------------------------------------------ #
        lb = np.asarray(func.bounds.lb, dtype=np.float64)
        ub = np.asarray(func.bounds.ub, dtype=np.float64)
        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            raise ValueError("func.bounds must contain `dim` lower and upper limits")
        if not np.allclose(lb, -5.0) or not np.allclose(ub, 5.0):
            raise ValueError("Expected bounds to be [-5, 5] in each dimension")

        # ------------------------------------------------------------------ #
        # 2. evaluate the provided initial design (counts against the budget)
        # ------------------------------------------------------------------ #
        n_init = self.initial_points.shape[0]
        if n_init > self.budget:
            raise ValueError("Number of initial points exceeds allowed budget")

        X_list = []          # torch tensors, shape (dim,)
        Y_list = []          # torch tensors, shape (1,)
        calls = 0
        best_f = np.inf
        best_x = None

        for x_np in self.initial_points:
            if calls >= self.budget:
                raise RuntimeError("Evaluation budget exhausted while processing initial points")
            f_val = float(func(x_np))
            calls += 1

            X_list.append(torch.from_numpy(x_np))
            Y_list.append(torch.tensor([f_val]))
            if f_val < best_f:
                best_f = f_val
                best_x = x_np.copy()

        # ------------------------------------------------------------------ #
        # 3. main BO loop (single‑point acquisition)
        # ------------------------------------------------------------------ #
        # Torch tensors for bounds used by BoTorch (unit cube after normalisation)
        bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])

        while calls < self.budget:
            # ----- 3.1. build / fit GP model -------------------------------------------------
            train_X = torch.stack(X_list)                     # (n, dim)
            train_Y = torch.stack(Y_list).view(-1, 1)         # (n, 1)

            # Normalise inputs to the unit hyper‑cube
            train_X_norm = normalize(train_X, torch.tensor(np.vstack([lb, ub]).T))

            # GP with a Matern‑5/2 kernel wrapped by a ScaleKernel; outcome is standardised
            gp = SingleTaskGP(
                train_X_norm,
                train_Y,
                outcome_transform=Standardize(m=1),
                covar_module=None,          # let the model pick the default (Matern)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # ----- 3.2. acquisition (Expected Improvement) ----------------------------------
            ei = ExpectedImprovement(
                model=gp,
                best_f=best_f,
                maximize=False,   # we minimise the objective
            )

            # ----- 3.3. optimise acquisition -----------------------------------------------
            # Use several random restarts + raw samples for robustness
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=500,
                options={"batch_limit": 5, "maxiter": 200},
            )  # candidate shape (1, dim) in the *normalised* space

            # ----- 3.4. map back to original space and evaluate ----------------------------
            x_new = unnormalize(candidate, torch.tensor(np.vstack([lb, ub]).T))
            x_np = x_new.squeeze(0).cpu().numpy()
            f_new = float(func(x_np))
            calls += 1

            # ----- 3.5. bookkeeping ---------------------------------------------------------
            X_list.append(candidate.squeeze(0))   # still normalised for next model fit
            Y_list.append(torch.tensor([f_new]))
            if f_new < best_f:
                best_f = f_new
                best_x = x_np.copy()

        return best_f, best_x