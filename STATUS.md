# Status
- Group 6, seeds 60-69, d=6. Targets: BBOB f21 (inst 1) + Practical 1125 Lennard-Jones (global min -1).
- LLM: gemini-3.1-flash-lite (Google AI Studio; key in .env, mode 600).
- Launched full LLaMEA searches (100 candidates each, tag run1), logs: results/llamea_logs/{bbob_f21,lj1125}_run1.out
  and results/llamea_logs/exp-*/ (log.jsonl, conversationlog.jsonl)
- Plain BoTorch BO reference (stage-1 eval): f21 AOCC 0.223, LJ AOCC 0.004.
- TODO: stage-2 benchmark script (CMA-ES, BoTorch BO, 2 generated algs; 2 problems x 10 seeds x 600 evals), plots, README.
- 14:00: scripts written: run_benchmark.py (resumable stage 2), make_plots.py, baselines/{cmaes,botorch_bo}.py.
  CMA-ES stage-2 done (results/stage2/*/cmaes_seed*.npy). BoTorch BO baseline running in background (2 procs; results/stage2_botorch.out).
  Waiting on LLaMEA searches; then: python run_benchmark.py --methods llamea_bbob_f21 llamea_lj1125 --gen bbob_f21=results/best_bbob_f21_run1.py --gen lj1125=results/best_lj1125_run1.py ; python make_plots.py ; README.
- 14:30: run1 of both searches stopped/archived (results/llamea_logs/archive_run1_crashed/, kept for the report).
  LJ run1 crashed at 14:17: joblib TimeoutError when a candidate exceeded LLaMEA's 20-min eval_timeout (unhandled by LLaMEA, no resume). f21 run1 killed pre-emptively (same risk).
  Fix: Evaluator enforces its own 1200s limit (SIGALRM) -> timeout = failed candidate with feedback; LLaMEA backstop timeout 7200s.
  Relaunched both as tag run2 (same prompts). LJ prompt change still undecided; revisit at ~candidate 30. Best code files: results/best_<target>_run2.py. make_plots.py now reads run2.
- 14:58: README drafted (sections 7-9 TODO after searches finish). requirements.txt frozen. 0 timeouts in first 24 candidates -> keep 1200s limit.
- 15:35: Added Cerebras (gpt-oss-120b, free tier; key in .env) via boa/llm.py RobustCerebras + run_llamea.py --llm cerebras --workers N. Decision (option 2): new Cerebras-only searches, tag cb1, 4 workers each, launched 15:34. Gemini run2 searches still running (kept alive on user's request; ~12 candidates each, slowed by Gemini 503s).
  Run metadata (provider/model/config) in results/llamea_logs/<target>_<tag>.meta.json. Candidate best files: results/best_<target>_<tag>.py
- 15:50: Cerebras abandoned (no longer free; logs in archive_cb1_gptoss/, archive_cb2_cerebras_qwen/, not used). Cerebras key removed from .env.
  User confirmed Mistral plan is free (Experiment). Launched Mistral-only searches codestral-latest, tag ms1, 5 workers each (results/llamea_logs/*_ms1.*, meta in <target>_ms1.meta.json).
  Gemini run2 searches still running in parallel. Final choice per target = best AOCC across runs; document all runs in README.
- 16:45: Mistral ms1 (codestral-latest) finished 100 candidates/target with 0 valid (outdated BoTorch API: fit_gpytorch_model etc.) -> archived (archive_ms1_codestral_0valid/), not usable.
  Free-plan quotas on this key: codestral 125 rpm/625k tpm; mistral-large-4 50 rpm/20k tpm; mistral-medium-3.5 & small = 0 rpm (unavailable).
  mistral-large-4 is a reasoning model: needs reasoning_effort (low/medium; high burns all tokens, 'none' unsupported). Using medium, max_tokens 12000. Pilot: 2/6 valid single-shot.
  Launched Mistral ms2 (mistral-large-4, medium) for both targets, 4 workers each. RobustMistral flattens list-type content (thinking/text chunks).
  Gemini run2: f21 28 logged (4 timeouts = 14%, best AOCC 0.249 at gen 3, beats plain BO 0.223); LJ 12 logged, stalled by 503s. BoTorch baseline 3/20.
- 17:15: Gemini run2: f21 36 logged/15 valid, best AOCC 0.261; LJ 36 logged/21 valid, best 0.107 (plain BO ref 0.004) -> Gemini is the strong run now.
  Mistral ms2 (mistral-large-4): f21 4 logged/0 valid, LJ 28 logged/2 valid (1.7e-7); heavy 429s from shared 20k tpm cap. Reniced ms2 processes (+10) so they do not slow Gemini runs (load avg 12-16 on 12 threads).
  BoTorch baseline 5/20 runs. Recommend stopping ms2 if it stays <10% valid; waiting for user input.
- 17:25: User asked to stop Mistral. ms2 stopped and archived (archive_ms2_mistral_large/: f21 4 logged/0 valid; LJ ~30 logged/2 valid at stop). Focus now: Gemini run2 searches + BoTorch baseline only.
  Provider history for README: Gemini run1 (crashed, timeout bug), Cerebras cb1/cb2 (no longer free, abandoned), Mistral ms1 codestral (0 valid), Mistral ms2 large-4 (very low yield). Final results come from Gemini run2.
- 17:45: Gemini run2 f21 52 logged/27 valid (best 0.261), LJ 60/39 (best 0.107); BoTorch baseline 6/20. Moved stray best_*_ms1/dry files to llamea_logs/archive_stray_best/. Rate after removing contention: ~0.5-0.8 candidates/min.
- 18:50: GEMINI DAILY QUOTA HIT: 500 req/day free tier for gemini-3.1-flash-lite exhausted (429 RESOURCE_EXHAUSTED, resets ~09:00 CET 2026-11-05). Searches (f21 68 logged best 0.277; LJ 76 logged best 0.107) are waiting in retry loop (wrapper retries up to 40x with <=450s sleeps ~ up to 3-4h, after which candidates fail with errors).
  Snapshot of current best saved as results/best_<target>_run2_provisional.py; Stage 2 for generated algs started on these (results/stage2_generated_provisional.out). If final best changes after search resumes/finishes, DELETE results/stage2/*/llamea_* and rerun with the final best files.
- 19:00: DECISION (user): option 1 — stop searches at the quota wall, use current results as final. Final searches: f21 run2 = 68 candidates evaluated (27+ valid; best AOCC 0.2774, gen 7, id bd2ca0ff, TuRBO-m + EI + ARD); LJ run2 = 76 evaluated (best AOCC 0.1071, gen 2, id dc040b7c, TuRBO-inspired local search + log-transformed GP).
  Files results/best_<target>_run2.py are now FINAL (identical to the provisional ones Stage 2 is already running on, so no recompute needed).
  Reason for stopping early (README): Gemini free-tier daily cap (500 req/day for gemini-3.1-flash-lite) hit at ~18:45; reset 09:00 next day; assignment allows <=100 candidates.
  TODO: wait for Stage 2 (generated algs + BoTorch baseline), make_plots.py, README sections 7-9 (results, component-by-component analysis + pseudocode of the two best algorithms, generalization discussion).
- 19:15: algorithm notes saved in results/algorithm_notes.md (incl. possible un-negated-Y issue in LJ algorithm to verify). Next wakeup 20:07 for Stage 2.
20:07 stage2: cmaes 20/20, botorch 15/20, llamea 3/40, no errors, 8 procs alive
20:08 generated job had crashed (missing _provisional.py path); relaunched with run2 paths, 6 procs -> results/stage2_resume.out
20:38 cmaes 20, botorch 15, llamea f21 7/10, others 0; 15 procs alive, load 16, no errors
21:19 llamea_bbob_f21 20/20 done, botorch 16/20, llamea_lj1125 0/20 (running), no errors
22:05 remaining: botorch lj 8/10, llamea_lj1125 on lj 5/10; rest done; no errors
22:46 79/80 done; only botorch lj one seed left
23:2x all 80 done, plots made, README 7-9 written, sign check in results/sign_check
