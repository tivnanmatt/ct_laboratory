# Scatter and off-focal models

Air-normalized measurement of pixel `j` for one firing, `t_j = y_j / I_0,j`, composed in
`physics/xray/xray_system/air_normalized_model.py` (`AirNormalizedProjectionModel`).
Paper notation: `ybar = G S_2 S_1 S_0 exp(-Q l)`. Each component below adds an incident fluence `phi` that is detected
through the same `S_2 S_1`.

| Component | Module | In gain scan? | In phantom scan | Parameters (prior) |
|---|---|---|---|---|
| Primary `P` | `TungstenSpectrum` + `ScintillatorDetector` | yes (`T = 1`) | `T_j = sum_E w_j(E) exp(-Q l)` | kVp ±1, Al 3 mm ×/÷2, W 20 µm ×/÷5, heel, 4-spline ±0.1, CsI 0.6 mm |
| Off-focal `G` | `OffFocalRadiation` | yes (`G = 1`) | Gaussian halo blur of `T` | `g` 10 % ± 5 %, halo sd 50 mm ×/÷2 (source plane) |
| Room `R` | `RoomScatter` | **yes**: `r F̄/F_j` | `r F̄/F_j [(1-h) + h K_R T]` | `r` 4.5 % ± 3 %, shadowed fraction `h` ~0.2 |
| Object `O` | `CentroidKleinNishinaScatter` | no | one Klein–Nishina point at the centroid | amplitude `alpha` 1 ×/÷3 (× physical) |

Physical model:

    t_j = [ (1 - g - r) P_j + g G_j + R_j + O_j ] / [ 1 - r + r F̄/F_j ]

The denominator is the gain scan: room scatter is in the air signal `I_0` too, so it is not simply added to the phantom
signal. `F̄/F_j` is the ratio of the mean to the pixel's smooth air fluence (beam pattern). The sharp detector efficiency
`eta_j` cancels, because every component is detected through the same `S_1,j`.

## Alternative: binned low-frequency additive scatter (`BinnedAdditiveScatter`)

Room and object scatter are replaced by an arbitrary smooth additive signal. It may be **different in the gain scan and
the phantom scan**:

    raw phantom  y_j   = I_0p,j T_j + s_ph,j
    raw gain     I_0,j = I_0p,j     + s_g,j        =>   t_j = (1 - a_g,j) [ (1-g) P_j + g G_j ] + a_ph,j

Here `a = s / I_0` is a fraction of the measured air signal. Each field is **flat over bins of 8 × 8 detector pixels**
(8 × 8 mm at 1 mm pitch), which gives **4 × 4 = 16 scatter points per 32 × 32 module**. Bins are indexed with
`module_bin_index(module, col, row, bin_px=8)`.

Free parameters per source:
- `a_g [n_bins]`, prior 4 % ± 3 %.
- `a_ph [n_fields, n_bins]`, prior 4 % ± 5 %. Use one phantom field per station, or one shared by stations that see the
  same object.
- An optional neighbour smoothness penalty `lambda sum (a_b - a_b')^2`, over neighbours within a module and across
  adjacent modules along the arc (`neighbour_pairs`).

In air only `a_ph - a_g` is identified. The split between the two fields comes from the variation of `T` inside a bin
and across stations, plus the priors. Use it as a model-agnostic check of the physical scatter model: if both models give
the same corrected primary, the physical model's shape is adequate.

## Measurement chain (`detector/measurement_chain.py`)

    raw = o_j + gamma_j N + eps,   N ~ Poisson(lambda),  eps ~ N(0, sigma_j^2)
    z = (raw - o)/gamma + kappa ~ Poisson(lambda + kappa),  kappa = sigma^2/gamma^2

- `ShiftedPoissonChain.nll`: the exact likelihood, used for reconstruction.
- `AirNormalizedGaussian`: the calibration likelihood. It is a Gaussian on `t` with variance
  `(t I_0 + kappa)/I_0^2 + floor^2 + (jitter |dt/dcol|)^2 + (dm/dL)^2 sigma_L^2`, so shadow edges and uncertain path
  lengths are down-weighted.

## Parameters (`physics/xray/parameters.py`)

Every model holds a `PriorParameters` set (`.params`). Values are stored whitened (`z = (u - mu)/sd` on the
unconstrained scale), so `neg_log_prior() = 0.5 sum z^2`. Parameters are fixed with `.fix(name)` and set with
`.set_value(name, v)`. The default priors are the 2026-10-01 spectral-cylinder calibration (120 kV); override them with
`set_prior`.

## Validation

`python_examples/test_xray_scatter_models.py` checks:
- `t = 1` in air for both models;
- the room scatter cancels in air;
- the binned layout (16 bins per module);
- gradients flow to every parameter;
- the KN factor is symmetric and ~0.03 % of air for a 30 mm water cylinder, as in the spectral-cylinder calibration.
