# gmi_merge_tests

Regression tests for the areas merged from the gmi repo into `ct_laboratory`:

- `linear_system/` — 29 modules ported from `gmi/tests/linear_system` (legacy
  multi-file LinearSystem stack). Includes hydra config-instantiation tests
  driven by the vendored `configs/linear_system/*.yaml` (targets rewritten to
  the `ct_laboratory` namespace).
- `sde/` — 8 modules ported from `gmi/tests/sde` (stochastic differential
  equations; depend on `ct_laboratory.linear_system`).
- `linalg/` — written fresh (gmi shipped none) for the consolidated
  `ct_laboratory.linalg` LinearSystem stack: forward/transpose/inverse
  contracts, adjoint consistency, Fourier unitarity.

`conftest.py` provides shared tensor fixtures (ported from gmi).

## Run

```bash
PYTHONPATH=/workspace/ct_laboratory \
  /opt/venv/bin/python -m pytest python_examples/gmi_merge_tests -q
```

All tests run on CPU (no GPU or CUDA extension required). Last run: 313 passed.
