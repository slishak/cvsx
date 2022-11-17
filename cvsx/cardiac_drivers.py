from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from cvsx import parameters as p


class CardiacDriverBase(ABC, eqx.Module):
    hr: float

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_wrapped = jnp.remainder(t, 60 / self.hr)
        return self.e(t_wrapped)

    @abstractmethod
    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        pass


class SimpleCardiacDriver(CardiacDriverBase):
    b: float

    def __init__(self, parameter_source: str = "smith"):

        params = p.cd_parameters[parameter_source]
        self.b = params["b"]
        self.hr = params["hr"]

    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.b * (t - 30 / self.hr) ** 2)


class GaussianCardiacDriver(CardiacDriverBase):
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray

    def __init__(self, parameter_source: str = "chung"):

        params = p.cd_parameters[parameter_source]
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]
        self.hr = params["hr"]

    def e(self, t: jnp.ndarray) -> jnp.ndarray:

        t_1d = jnp.atleast_1d(t)
        f = lambda t_i: jnp.sum(self.a * jnp.exp(-self.b * (t_i - self.c) ** 2))
        f_v = jax.vmap(f, 0, 0)
        e_t = f_v(t_1d)

        return jnp.reshape(e_t, t.shape)
