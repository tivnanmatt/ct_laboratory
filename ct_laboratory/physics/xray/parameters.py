"""Physical parameters with Gaussian priors for the X-ray measurement models.

Every scalar or vector parameter is stored WHITENED, z = (u - mu) / sd, where u is the unconstrained value and the
physical value is ``transform(u)``.  The negative log prior is then simply 0.5 * sum(z^2), first-order optimizers
see well-scaled gradients, and freezing a parameter is ``requires_grad_(False)``.

Transforms:
    'identity'  value = u
    'exp'       value = exp(u)          (positive quantities; prior is log-normal, sd is a log-ratio)
    'sigmoid'   value = sigmoid(u)      (fractions in (0, 1); prior is logit-normal)

Example:
    p = PriorParameters()
    p.add('al_mm', 17.0, sd=math.log(2.0), transform='exp')     # prior median 17 mm, factor-2 sd
    p['al_mm']                                                     # -> tensor(17.)
    p.neg_log_prior()                                              # -> 0 at the prior mean
"""
import math
import torch


_FWD = {
    'identity': lambda u: u,
    'exp': torch.exp,
    'sigmoid': torch.sigmoid,
}
_INV = {
    'identity': lambda v: v,
    'exp': math.log,
    'sigmoid': lambda v: math.log(v / (1.0 - v)),
}


class PriorParameters(torch.nn.Module):
    """A named set of whitened parameters with independent Gaussian priors on the unconstrained scale."""

    def __init__(self):
        super().__init__()
        self.z = torch.nn.ParameterDict()
        self._meta = {}

    def add(self, name, value, sd, transform='identity', shape=(), free=True, label=None, display=None):
        """Register ``name`` with prior centred on the physical ``value`` (prior sd ``sd`` on the unconstrained scale).

        ``shape`` makes a vector parameter sharing one prior (e.g. per-bin scatter amplitudes).
        ``display`` optionally maps the physical value to a printable number (e.g. lambda v: 100 * v for %).
        """
        mu = _INV[transform](float(value)) if not torch.is_tensor(value) else value
        self._meta[name] = dict(mu=mu, sd=float(sd), transform=transform, label=label or name, display=display)
        self.z[name] = torch.nn.Parameter(torch.zeros(shape), requires_grad=free)
        return self

    def set_prior(self, name, value=None, sd=None):
        m = self._meta[name]
        if value is not None:
            m['mu'] = _INV[m['transform']](float(value)) if not torch.is_tensor(value) else value
        if sd is not None:
            m['sd'] = float(sd)
        return self

    def set_value(self, name, value):
        """Move the current value (not the prior) to the physical ``value``."""
        m = self._meta[name]
        u = torch.as_tensor(_INV[m['transform']](float(value)) if not torch.is_tensor(value) else value)
        with torch.no_grad():
            self.z[name].copy_(((u - torch.as_tensor(m['mu'])) / m['sd']).expand_as(self.z[name]))
        return self

    def free(self, *names, free=True):
        for n in (names or self.z.keys()):
            self.z[n].requires_grad_(free)
        return self

    def fix(self, *names):
        return self.free(*names, free=False)

    def unconstrained(self, name):
        m = self._meta[name]
        mu = torch.as_tensor(m['mu'], device=self.z[name].device, dtype=self.z[name].dtype)
        return mu + m['sd'] * self.z[name].clamp(-12.0, 12.0)

    def __getitem__(self, name):
        return _FWD[self._meta[name]['transform']](self.unconstrained(name))

    def names(self, free_only=False):
        return [n for n in self.z.keys() if (self.z[n].requires_grad or not free_only)]

    def neg_log_prior(self):
        """0.5 * sum z^2 over the FREE parameters (fixed ones sit at their set value and carry no prior term)."""
        tot = 0.0
        for n, z in self.z.items():
            if z.requires_grad:
                tot = tot + 0.5 * (z.double() ** 2).sum()
        return tot

    def summary(self):
        """{label: printable value} for scalar parameters, {label: (mean, min, max)} for vectors."""
        out = {}
        with torch.no_grad():
            for n in self.z.keys():
                m = self._meta[n]; v = self[n].detach().cpu()
                if m['display'] is not None:
                    v = m['display'](v)
                out[m['label']] = float(v) if v.numel() == 1 else (float(v.mean()), float(v.min()), float(v.max()))
        return out

    def whitened(self):
        """{label: z} (prior-sd units) of the scalar parameters, for parameter-trajectory plots."""
        return {self._meta[n]['label']: float(z.detach()) for n, z in self.z.items() if z.numel() == 1}
