import equinox as eqx
import jax.numpy as jnp

from cvsx.unit_conversions import convert


class RespiratoryPatternGenerator(eqx.Module):
    hb: float = convert(1.0, "1/l")
    a: float = -0.8
    b: float = -3.0
    alpha: float = 1.0
    lam: float = convert(1.5, "mmHg")
    mu: float = convert(1.0, "mmHg")
    beta: float = 0.1

    def __call__(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        args=tuple[jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Derivatives of central respiratory pattern generator.

        Args:
            t (jnp.ndarray): Time (s)
            states (jnp.ndarray):  ODE states formatted as dict
            args (jnp.ndarray): dv_alv_dt input. Defaults to 0.0.

        Returns:
            dict[str, jnp.ndarray]:  ODE derivatives formatted as dict
        """
        dv_alv_dt = args[0]
        dx_dt = self.alpha * (self.lienard(states["x"], states["y"]) - self.hb * dv_alv_dt)
        dy_dt = self.alpha * states["x"]
        dp_mus_dt = self.lam * states["y"] + self.mu - self.beta * states["p_mus"]

        return {
            "x": dx_dt,
            "y": dy_dt,
            "p_mus": dp_mus_dt,
        }

    def lienard(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Lienard function.

        Args:
            x (jnp.ndarray): x value
            y (jnp.ndarray): y value

        Returns:
            jnp.ndarray: f(x, y)
        """
        f = (self.a * y**2 + self.b * y) * (x + y)
        return f


class PassiveRespiratorySystem(eqx.Module):
    e_alv: float = convert(5, "cmH2O/l")
    e_cw: float = convert(4, "cmH2O/l")
    r_ua: float = convert(5, "cmH2O s/l")
    r_ca: float = convert(1, "cmH2O s/l")
    v_th0: float = convert(2, "l")

    def __call__(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        return_outputs: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Passive respiratory model implementation.

        Args:
            t (jnp.ndarray): Time (s)
            states (jnp.ndarray): Model states

        Returns:
            dict[str, jnp.ndarray]: Model outputs
        """

        # These are not states of this model, but states of the overall system
        v_pcd = states["v_lv"] + states["v_rv"]
        v_pu = states["v_pu"]
        v_pa = states["v_pa"]

        v_bth = v_pcd + v_pu + v_pa
        v_th = v_bth + states["v_alv"]
        p_pl = states["p_mus"] + self.e_cw * (v_th - self.v_th0)

        dv_alv_dt = -(p_pl + self.e_alv * states["v_alv"]) / (self.r_ca + self.r_ua)

        derivatives = {
            "v_alv": dv_alv_dt,
        }

        if not return_outputs:
            return derivatives

        outputs = {
            "p_pl": p_pl,
            "v_th": v_th,
            "v_bth": v_bth,
        }

        return derivatives, outputs
