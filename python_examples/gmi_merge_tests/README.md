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
- `diffusion/` — dataset-free smoke tests for the merged diffusion stack
  (`diffusion` + `sde` + `random_variable_gmi` + `samplers`).
- `network/` — written fresh (gmi shipped no network tests) for the merged
  `ct_laboratory.network` area: native `SimpleCNN` (2D and 3D — the 3D path was
  broken in gmi and is fixed here), `DenseNet`, `LinearConv`, `LambdaLayer`, and
  the new native `ConfigurableUNet` (2D/3D forward+backward, timestep embedding,
  odd group counts). Diffusers-backed 2D U-Nets are tested via
  `pytest.importorskip("diffusers")`.
- `test_loss_lr.py` — `loss_function.inv_t_weighted_mse` (guards the current
  t-ignored / plain-MSE behavior) and `lr_scheduler.LinearWarmupLRScheduler`.

`conftest.py` provides shared tensor fixtures (ported from gmi).

## Run

```bash
PYTHONPATH=/workspace/ct_laboratory \
  /opt/venv/bin/python -m pytest python_examples/gmi_merge_tests -q
```

All tests run on CPU (no GPU or CUDA extension required). Last run: 345 passed
(diffusers-backed network tests are skipped when `diffusers` is not installed).
