# Contributing

Use a small, focused change and describe the observable behaviour it improves. Discuss larger solver changes in an issue first.

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
```

Numerical changes need independent feasibility checks. Keep the allocation variables, rate normalisation, decoding order and forwarding model consistent. A solver status is not a substitute for checking the returned point. Keep candidate construction and all data-dependent starts inside the declared runtime budget.

Use new output folders for each experiment. Report processor and library versions when comparing timings. Do not commit generated run directories, credentials, local environment folders or third-party solver binaries.
