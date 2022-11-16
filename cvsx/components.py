import equinox as eqx
import jax.numpy as jnp


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
