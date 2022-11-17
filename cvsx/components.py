import equinox as eqx
import jax.numpy as jnp

from cvsx.unit_conversions import convert


class PressureVolume(eqx.Module):
    e: float
    v_d: float
    v_0: float
    lam: float
    p_0: float

    def p(self, v: jnp.ndarray, e_t: jnp.ndarray) -> jnp.ndarray:
        p_es = self.p_es(v)
        p_ed = self.p_ed(v)
        return e_t * p_es + (1 - e_t) * p_ed

    def p_es(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.e * (v - self.v_d)

    def p_ed(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.p_0 * (jnp.exp(self.lam * (v - self.v_0)) - 1)

    def p_ed_linear(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.p_0 * self.lam * (v - self.v_0)


class BloodVessel(eqx.Module):
    r: float

    def flow_rate(self, p_upstream: jnp.ndarray, p_downstream: jnp.ndarray) -> jnp.ndarray:
        q_flow = (p_upstream - p_downstream) / self.r
        return q_flow


class Valve(BloodVessel):
    def open(
        self,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        return p_upstream > p_downstream

    def flow_rate(self, p_upstream: jnp.ndarray, p_downstream: jnp.ndarray) -> jnp.ndarray:
        q_flow = super().flow_rate(p_upstream, p_downstream)
        return jnp.where(self.open(p_upstream, p_downstream, q_flow), q_flow, 0.0)


class InertialValve(Valve):
    l: float

    def open(
        self,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.logical_or(p_upstream > p_downstream, q_flow > 0.0)

    def flow_rate_deriv(
        self,
        p_upstream: jnp.ndarray,
        p_downstream: jnp.ndarray,
        q_flow: jnp.ndarray,
    ) -> jnp.ndarray:
        dq_dt = (p_upstream - p_downstream - q_flow * self.r) / self.l
        return jnp.where(self.open(p_upstream, p_downstream, q_flow), dq_dt, 0.0)


class SmoothInertialValve(InertialValve):
    q_threshold_slope: float = convert(10.0, "ml/s")
    q_threshold_min: float = convert(10.0, "ml/s")

    def flow_rate_deriv(
        self,
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
