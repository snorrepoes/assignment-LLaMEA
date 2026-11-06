# Notes for README section 8 (verify before final write-up)
## f21 algorithm (results/best_bbob_f21_run2.py, AOCC 0.277, gen 7)
- Description claims "TuRBO-m (multi trust region)"; code has ONE trust region (box of half-width `length` around the incumbent, in normalised space). Say what the code does.
- GP: SingleTaskGP, ScaleKernel(Matern-2.5, ARD), GaussianLikelihood; inputs normalised manually to [0,1]^d; raw y (no log/standardise explicit).
- Acquisition: analytic EI (maximize=False) optimised inside the trust region (10 restarts, 128 raw samples); q=1.
- TR rule: length 0.5 start; x1.1 (max 1.0) on improvement, x0.9 (min 0.05) otherwise. Re-fits GP every iteration. Init = given 12 Sobol points.
## LJ algorithm (results/best_lj1125_run2.py, AOCC 0.107, gen 2)
- Output transform: Y = log(y - min(y) + 1e-9) then Standardize; GP default kernel (BoTorch default), Normalize inputs.
- Acquisition: qNoisyEI (X_baseline = all X) inside trust region of half-width `length` (in ORIGINAL units, box [-5,5]; start 1.0, x1.1 up to 2.0 on improvement, x0.8 down to 0.05 otherwise), 10 restarts, 256 raw samples.
- POSSIBLE BUG TO VERIFY EXPERIMENTALLY: the log-transformed Y is NOT negated, yet BoTorch acquisitions maximise; so qNEI seems to look for HIGH log-regret (worse) points inside the trust region. Check with a variant using -Y on a few seeds (CPU is busy now) before claiming; if true, explain the score as exploration/space-filling effect around the incumbent + shrinking TR, and flag in generalization discussion.
- Also: best_f changes every iteration since transform is relative to current min.
