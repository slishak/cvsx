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
            raise RuntimeError("Inertial valve has no flow_rate_deriv method")
        dq_dt = (p_upstream - p_downstream - q_flow * self.r) / self.l
        return dq_dt


class Valve(BloodVessel):
    allow_reverse_flow: bool = False

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
    allow_reverse_flow: bool = True
    r_reverse: float = convert(10, "mmHg/ml")

    def open(
        self,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.inertial:
            return q_flow > 0.0
        else:
            return p_upstream > p_downstream

    def flow_rate_reversed(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
    ) -> jnp.ndarray:
        return (p_upstream - p_downstream) / self.r_reverse

    def flow_rate_deriv_reversed(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        return (p_upstream - p_downstream - q_flow * self.r_reverse) / self.l

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

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        dq_dt_fwd = super().flow_rate_deriv(t, p_upstream, p_downstream, q_flow)
        dq_dt_rev = self.flow_rate_deriv_reversed(t, p_upstream, p_downstream, q_flow)
        # return jnp.maximum(dq_dt_fwd, dq_dt_rev)
        return jnp.where(self.open(p_upstream, p_downstream, q_flow), dq_dt_fwd, dq_dt_rev)


class SmoothValve(TwoWayValve):
    q_threshold_slope: float = convert(10.0, "ml/s")
    q_threshold_min: float = convert(10.0, "ml/s")

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        # Differential pressure across valve
        dp = p_upstream - p_downstream
        # Find threshold flow rate below which to smooth out
        q_threshold = self.q_threshold_slope * -dp / convert(1.0, "mmHg")
        q_threshold = jnp.where(
            q_threshold > self.q_threshold_min, q_threshold, self.q_threshold_min
        )

        # Flow rate derivative
        dq_dt = (dp - q_flow * self.r) / self.l

        # Jacobian element d(dq_dt)_dq at threshold point, to match gradient
        dq_dt_at_q_threshold = (dp - q_threshold * self.r) / self.l

        # Quadratic smoothing between threshold and (0, 0)
        a = -self.r / (self.l * q_threshold) - dq_dt_at_q_threshold / q_threshold**2
        b = 2 * dq_dt_at_q_threshold / q_threshold + self.r / self.l
        dq_dt_smooth = a * q_flow**2 + b * q_flow

        dq_dt = jnp.where((dp > 0) | (q_flow > q_threshold), dq_dt, dq_dt_smooth)

        return dq_dt
