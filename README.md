# py-gravity

Research experiments with mutual information: a cellular automaton, local attention for synthetic time series, and an optional GPT-2 attention experiment.

## Run locally

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python info_gravity_ca.py --help
uv run python bench_timeseries.py --device cpu --epochs 1 --series 2 --T 64 --ctx 8 --bs 8 --win 3 --d_model 8 --d_ff 16 --layers 1 --bins 4 --outdir runs/smoke
```

See `python info_gravity_ca.py --help` for grid, step and output settings. Outputs go into local run folders and are ignored by Git.

## Stack and limits

NumPy, SciPy, Matplotlib and PyTorch. Inputs are synthetic. The models are experiments, not evidence for a physical theory or a validated performance improvement. CPU execution needs no external account; MPS or CUDA is optional.

`experiments/test_mi_nlp.py` is an optional benchmark, not a unit test. Install its dependencies with `uv sync --extra nlp`. It downloads public GPT-2 weights and allocates a dense vocabulary matrix, which can exceed 10 GB. It was not run as part of the local smoke check. Its Transformers version is pinned for the older attention API. There is no automated unit-test suite.

Licensed under Apache-2.0. See [LICENSE](LICENSE).
