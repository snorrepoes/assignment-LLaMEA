# assignment-LLaMEA

Exercises on Bayesian Optimization and LLM-driven algorithm generation with
[LLaMEA](https://github.com/XAI-liacs/LLaMEA), `ioh`, and `botorch`.

## Setup with uv

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python
version and dependencies.

1. Install `uv` (if you don't have it already):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install the project's dependencies into a local virtual environment:

   ```bash
   uv sync
   ```

   This reads `pyproject.toml` / `uv.lock`, creates `.venv/`, and installs
   everything pinned there (including `llamea`, `ioh`, and `botorch`).

3. Configure your API key. Copy the sample env file and fill in your key:

   ```bash
   cp .env.sample .env
   ```

   Then edit `.env` and set `OPENAI_KEY` to your OpenAI (or OpenAI-compatible)
   API key. `.env` is git-ignored and is loaded automatically at runtime via
   [`python-dotenv`](https://pypi.org/project/python-dotenv/) — never commit
   it or paste a real key into the notebook or scripts.

## Running the notebook

```bash
uv run jupyter notebook LLaMEA_Working_Groups_students.ipynb
```

`uv run` executes the command inside the project's virtual environment
without needing to activate it manually.

## Minimal example

`examples/minimal_llamea_bo.py` is a small, isolated script distilled from
the notebook. It:

1. Defines the `AOCCLogger` / `correct_aocc` helpers used to score an
   algorithm on a BBOB problem (via `ioh`) with the AOCC metric.
2. Runs a single LLaMEA outer-loop generation that asks the LLM to generate
   a `Optimize_with_BO` class, then evaluates it with the AOCC metric.

Run it with:

```bash
uv run examples/minimal_llamea_bo.py
```

It loads `OPENAI_KEY` from `.env` the same way the notebook does — make sure
you've completed step 3 above first. This makes one real call to the
configured LLM, so running it will use your API quota/credits.