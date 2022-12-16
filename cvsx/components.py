from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp

from cvsx.unit_conversions import convert


class PressureVolume(eqx.Module):
    _e: float
    v_d: float
    v_0: float
    _lam: float
    p_0: float

    _e_scale: float = 100.0
    _lam_scale: float = 10.0

    def __init__(self, e, v_d, v_0, lam, p_0, e_scale=100.0, lam_scale=10.0):
        self._e = jnp.sqrt(e / self._e_scale)
        self.v_d = v_d
        self.v_0 = v_0
        self._lam = jnp.sqrt(lam / self._lam_scale)
        self.p_0 = p_0
        self._e_scale = e_scale
        self._lam_scale = lam_scale

    @property
    def e(self):
        return self._e**2 * self._e_scale

    @property
    def lam(self):
        return self._lam**2 * self._lam_scale

    def p(self, t: jnp.ndarray, v: jnp.ndarray, e_t: jnp.ndarray) -> jnp.ndarray:
        p_es = self.p_es(t, v)
        p_ed = self.p_ed(t, v)
        return e_t * p_es + (1 - e_t) * p_ed

    def p_es(self, t: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return self.e * (v - self.v_d)

    def p_ed(self, t: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return self.p_0 * (jnp.exp(self.lam * (v - self.v_0)) - 1)

    def p_ed_linear(self, t: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return self.p_0 * self.lam * (v - self.v_0)


class BloodVessel(eqx.Module):
    _r: float
    _l: float
    inertial: bool

    def __init__(self, r, l=0.0, inertial=False):
        self._r = jnp.sqrt(r)
        self._l = jnp.sqrt(l)
        self.inertial = inertial

    @property
    def r(self):
        return self._r**2

    @property
    def l(self):
        return self._l**2

    def flow_rate(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
    ) -> jnp.ndarray:
        q_flow = (p_upstream - p_downstream) / self.r
        return q_flow

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        if not self.inertial:
            raise RuntimeError("Non-inertial valve has no flow_rate_deriv method")
        dq_dt = (p_upstream - p_downstream - q_flow * self.r) / self.l
        return dq_dt


class Valve(BloodVessel):
    allow_reverse_flow: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_reverse_flow = False

    def open(
        self,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.inertial:
            return jnp.logical_or(p_upstream > p_downstream, q_flow > 0.0)
        else:
            return p_upstream > p_downstream

    def flow_rate(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
    ) -> jnp.ndarray:
        q_flow = super().flow_rate(t, p_upstream, p_downstream)

        # Regardless of inertial valve or not, ignore inertia and consider steady state
        valve_open = p_upstream > p_downstream

        return jnp.where(valve_open, q_flow, 0.0)

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        dq_dt = super().flow_rate_deriv(t, p_upstream, p_downstream, q_flow)
        valve_open = self.open(p_upstream, p_downstream, q_flow)
        return jnp.where(valve_open, dq_dt, 0.0)


class TwoWayValve(Valve):
    r_reverse: float
    method: str

    def __init__(
        self,
        *args,
        r_reverse=convert(1, "mmHg/ml"),
        method="regurgitating",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.allow_reverse_flow = True
        self.r_reverse = r_reverse
        if method not in ("restoring", "restoring_continuous", "regurgitating"):
            raise NotImplementedError(method)
        if not self.inertial and method != "regurgitating":
            raise NotImplementedError("Restoring non-inertial valve")
        self.method = method

    def open(
        self,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:

        if not self.inertial:
            return p_upstream > p_downstream

        if self.method == "regurgitating":
            return q_flow > 0.0

        return super().open(p_upstream, p_downstream, q_flow)

    def flow_rate(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
    ) -> jnp.ndarray:
        q_flow = super().flow_rate(t, p_upstream, p_downstream)
        q_flow_rev = self.flow_rate_reversed(t, p_upstream, p_downstream)

        # Regardless of inertial valve or not, ignore inertia and consider steady state
        valve_open = p_upstream > p_downstream

        return jnp.where(valve_open, q_flow, q_flow_rev)

    def flow_rate_reversed(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.method == "regurgitating":
            return (p_upstream - p_downstream) / self.r_reverse
        else:
            return jnp.zeros_like(p_upstream)

    def flow_rate_deriv_reversed(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.method == "regurgitating":
            return (p_upstream - p_downstream - q_flow * self.r_reverse) / self.l
        else:
            return (-q_flow * self.r_reverse) / self.l

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        dq_dt_fwd = super().flow_rate_deriv(t, p_upstream, p_downstream, q_flow)
        dq_dt_rev = self.flow_rate_deriv_reversed(t, p_upstream, p_downstream, q_flow)
        if self.method == "restoring_continuous":
            return jnp.maximum(dq_dt_fwd, dq_dt_rev)
        else:
            return jnp.where(self.open(p_upstream, p_downstream, q_flow), dq_dt_fwd, dq_dt_rev)


class SmoothValve(Valve):
    r_reverse: float

    def __init__(
        self,
        r: float,
        l: float,
        inertial: bool = True,
        r_reverse: float = convert(1, "mmHg/ml"),
    ):
        super().__init__(r, l, inertial=inertial)
        self.allow_reverse_flow = True
        # Prevent r_reverse from being too low
        r_reverse = jnp.where(r_reverse < 2 * r, 2 * r, r_reverse)
        self.r_reverse = r_reverse

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        # Differential pressure across valve
        dp = p_upstream - p_downstream

        # Flow rate derivative
        dq_dt = (dp - q_flow * self.r) / self.l

        # Quadratic coefficients satisfying constraints:
        #   dq_dt_smooth = a*q**2 + b*q
        #   Equal to zero at q=0
        #   Gradient at q=0 is -r_reverse/l
        #   Matches both value and gradient of dq_dt at some q_threshold
        a = -((self.r - self.r_reverse) ** 2) / (4 * self.l * dp)
        q_threshold = 2 * dp / (self.r - self.r_reverse)
        dq_dt_smooth = a * q_flow**2 - self.r_reverse * q_flow / self.l

        # Use smooth version below q_threshold
        dq_dt = jnp.where((dp > 0) | (q_flow > q_threshold), dq_dt, dq_dt_smooth)

        # When q<0, use linear extrapolation
        dq_dt = jnp.where((dp > 0) | (q_flow > 0), dq_dt, -self.r_reverse * q_flow / self.l)

        return dq_dt
