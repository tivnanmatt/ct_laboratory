"""Air-normalized polyenergetic projection model for one firing:  t_j = y_j / I_0,j  as a function of the object line integrals.

Paper notation (ybar = G S_2 S_1 S_0 exp(-Q l)) extended with off-focal radiation, room scatter and object scatter:

    w_j(E)  = S_0(E; cone_j) [S_2 S_1]_j(E) / sum_E (...)              normalized detected spectrum of pixel j
    T_j     = < sum_E w_j(E) exp(-[Q l]_j(E)) >_sub-rays                  primary transmission (polyenergetic, pixel-averaged)
    P_j     = (K_focal T)_j                                               focal-spot blur (optional)
    G_j     = (K_G T)_j                                                   off-focal halo           (OffFocalRadiation)

  scatter = 'physical':
    t_j = [ (1 - g - r) P_j + g G_j + R_j + O_j ] / [ 1 - r + r F_bar/F_j ]
          R_j  room scatter, also in the gain scan -> denominator           (RoomScatter)
          O_j  object scatter, one Klein-Nishina point at the centroid      (CentroidKleinNishinaScatter)
  scatter = 'binned':
    t_j = (1 - a_g,j) [ (1 - g) P_j + g G_j ] + a_ph,j                       (BinnedAdditiveScatter, 8 x 8 mm flat bins)

Check (air, l = 0): physical -> [(1-g-r) + g + r F_bar/F] / [1 - r + r F_bar/F] = 1;  binned -> 1 - a_g + a_ph (= 1 when the
gain- and phantom-scan scatter are equal, as they must be in air).

Inputs per call (one firing, image layout [n_rows, n_cols] with columns ordered along the arc, see scatter.kernels):
    Ql          [n_rows, n_cols, n_sub, n_E]  attenuation exponent per sub-ray and energy, e.g. L[..., None] * mu_water
    cos_inc     [n_rows, n_cols]              incidence cosine at the detector
    cone        [n_rows, n_cols]              cone angle (deg) relative to the central plane (heel effect)
    fluence_ratio [n_rows, n_cols]            F_bar / F_j of the smooth air fluence (room scatter); ones if unknown
    obj_factors (F, D_air)                    from object_scatter.geometry_factors (physical model only)
    arc_mm [n_cols], row_mm [n_rows], magnification (detector / source-plane scale through the object)
"""
import torch


class AirNormalizedProjectionModel(torch.nn.Module):
    def __init__(self, source, detector, off_focal, room=None, object_scatter=None, binned=None, focal_blur=None):
        super().__init__()
        self.source, self.detector, self.off_focal = source, detector, off_focal
        self.room, self.object_scatter, self.binned, self.focal_blur = room, object_scatter, binned, focal_blur
        self.scatter = 'binned' if binned is not None else 'physical'

    def parameter_sets(self):
        """{component: PriorParameters} of all components present"""
        out = dict(source=self.source.params, detector=self.detector.params, off_focal=self.off_focal.params)
        for k in ('room', 'object_scatter', 'binned'):
            m = getattr(self, k)
            if m is not None:
                out[k] = m.params
        return out

    def neg_log_prior(self):
        tot = sum(p.neg_log_prior() for p in self.parameter_sets().values())
        if self.binned is not None:
            tot = tot + self.binned.penalty()
        return tot

    def spectra(self, cos_inc, cone):
        """normalized detected spectrum w [n_rows, n_cols, n_E]"""
        w = self.source(cone) * self.detector.detected_weight(cos_inc)
        return w / w.sum(-1, keepdim=True).clamp(min=1e-30)

    def forward(self, Ql, cos_inc, cone, arc_mm, row_mm, magnification, fluence_ratio=None, obj_factors=None,
                density_scale=1.0, phantom_field=0):
        w = self.spectra(cos_inc, cone)
        TE = torch.exp(-Ql).mean(-2)                                       # [R, C, E] sub-ray averaged transmission
        T = (TE * w).sum(-1)
        P = self.focal_blur(T, arc_mm, row_mm, magnification) if self.focal_blur is not None else T
        G = self.off_focal(T, arc_mm, row_mm, magnification)
        g = self.off_focal.fraction
        out = dict(w=w, TE=TE, T=T)
        if self.scatter == 'binned':
            pre = (1 - g) * P + g * G
            ag = self.binned.gain_field(); aph = self.binned.phantom_field(phantom_field)
            out.update(t=(1 - ag) * pre + aph, primary=(1 - ag) * (1 - g) * P, off_focal=(1 - ag) * g * G,
                       scatter=aph, gain_scatter=ag, denominator=torch.ones_like(T), primary_weight=(1 - ag) * (1 - g))
            return out
        fr = torch.ones_like(T) if fluence_ratio is None else fluence_ratio
        r = self.room.fraction if self.room is not None else torch.zeros((), device=T.device)
        R = self.room(T, fr, arc_mm, row_mm, magnification) if self.room is not None else torch.zeros_like(T)
        if self.object_scatter is not None and obj_factors is not None:
            O = self.object_scatter(self.source(), obj_factors, density_scale)
        else:
            O = torch.zeros_like(T)
        den = 1 - r + r * fr
        out.update(t=((1 - g - r) * P + g * G + R + O) / den, primary=(1 - g - r) * P / den, off_focal=g * G / den,
                   room=R / den, object=O / den, scatter=(R + O) / den, denominator=den, primary_weight=(1 - g - r) / den)
        return out
