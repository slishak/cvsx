from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from cvsx import parameters as p


class CardiacDriverBase(ABC, eqx.Module):
    hr: Union[float, Callable]
    dynamic: bool = False

    def __init__(self, hr: Union[float, Callable]):
        self.hr = hr
        self.dynamic = callable(hr)

    def __call__(self, t: jnp.ndarray, s: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if self.dynamic:
            ds_dt = self.hr(t) / 60
            s_wrapped = jnp.remainder(s, 1)
            e_s = self.e(s_wrapped)
            return e_s, ds_dt
        else:
            t_wrapped = jnp.remainder(t, 60 / self.hr)
            e_t = self.e(t_wrapped)
            return e_t

    @abstractmethod
    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        pass


class SimpleCardiacDriver(CardiacDriverBase):
    b: float

    def __init__(
        self,
        parameter_source: str = "smith",
        hr: Optional[Union[float, Callable]] = None,
    ):
        params = p.cd_parameters[parameter_source]
        self.b = params["b"]
        if hr is None:
            hr = params["hr"]
        super().__init__(hr)

    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        if self.dynamic:
            hr = self.hr(t)
        else:
            hr = self.hr
        return jnp.exp(-self.b * (t - 30 / hr) ** 2)


class GaussianCardiacDriver(CardiacDriverBase):
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray

    def __init__(
        self,
        parameter_source: str = "chung",
        hr: Optional[Union[float, Callable]] = None,
    ):

        params = p.cd_parameters[parameter_source]
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]
        if hr is None:
            hr = params["hr"]
        super().__init__(hr)

    def e(self, t: jnp.ndarray) -> jnp.ndarray:

        t_1d = jnp.atleast_1d(t)
        f = lambda t_i: jnp.sum(self.a * jnp.exp(-self.b * (t_i - self.c) ** 2))
        f_v = jax.vmap(f, 0, 0)
        e_t = f_v(t_1d)

        return jnp.reshape(e_t, t.shape)
