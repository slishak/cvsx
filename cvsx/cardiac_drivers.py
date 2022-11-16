from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp


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

    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.b * (t - 30 / self.hr) ** 2)


class GaussianCardiacDriver(CardiacDriverBase):
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray

    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(self.a * jnp.exp(-self.b * (t - self.c) ** 2))
