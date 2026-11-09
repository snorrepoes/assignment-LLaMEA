"""CMA-ES baseline (pycma: https://github.com/CMA-ES/pycma, Hansen et al. 2019, DOI 10.5281/zenodo.2559634).
Same protocol as every method: the shared Sobol initial design is evaluated first (counts against the
budget), then CMA-ES starts from the best initial point with sigma0 = 2 (20% of the box width)."""
import numpy as np, cma


class Optimize_with_CMAES:
    def __init__(self, budget, dim, initial_points, random_seed, sigma0=2.0):
        self.budget, self.dim, self.seed, self.sigma0 = budget, dim, random_seed, sigma0
        self.X0 = np.asarray(initial_points, float)

    def __call__(self, func):
        X = list(self.X0[: self.budget]); y = [func(x) for x in X]
        x0 = X[int(np.argmin(y))]
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, {"bounds": [func.bounds.lb.tolist(), func.bounds.ub.tolist()],
                                      "seed": self.seed + 1, "verbose": -9, "maxfevals": self.budget - len(y)})
        from boa.problems import BudgetExhausted
        try:
            while not es.stop() and len(func.values) < self.budget:
                xs = es.ask(); es.tell(xs, [func(x) for x in xs])
        except BudgetExhausted:
            pass
        i = int(np.argmin(func.values)); return func.values[i], None
