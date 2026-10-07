"""Detector-plane smoothing kernels shared by the scatter / off-focal models.

Layout convention for one firing (one source, its active modules): images ``X[..., n_rows, n_cols]`` where the columns are
ordered along the detector arc with positions ``arc_mm[n_cols]`` (module gaps are therefore handled exactly) and the rows have
positions ``row_mm[n_rows]``.

Blurs use the "air outside" boundary:  K X := 1 + K_r (X - 1) K_c^T  with row-normalized Gaussians, i.e. the signal beyond the
active modules is taken to be unattenuated air.  This is the right boundary for transmission-like quantities (off-focal halo,
room-scatter shadow), whose values tend to 1 away from the object.
"""
import torch


def gaussian_matrix(pos_mm, sigma_mm):
    """[n, n] row-normalized Gaussian weights between positions ``pos_mm`` (sigma may be a 0-d tensor)."""
    d = pos_mm[:, None] - pos_mm[None, :]
    k = torch.exp(-0.5 * (d / sigma_mm) ** 2)
    return k / k.sum(1, keepdim=True)


def blur_air_padded(X, arc_mm, row_mm, sigma_mm):
    """1 + K_r (X - 1) K_c^T for X [..., n_rows, n_cols] (separable Gaussian, sd ``sigma_mm`` at the detector)."""
    Kc = gaussian_matrix(arc_mm, sigma_mm)
    Kr = gaussian_matrix(row_mm, sigma_mm)
    return 1.0 + Kr @ (X - 1.0) @ Kc.T
