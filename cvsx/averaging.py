from typing import Callable

import equinox as eqx
import jax.numpy as jnp


class MovingAverage(eqx.Module):
    name: str
    time_constant: float

    def __call__(self, state: jnp.ndarray, outputs: dict) -> jnp.ndarray:
        return (outputs[self.name] - state) / self.time_constant


class GatedMovingAverage(MovingAverage):
    condition: Callable

    def __call__(self, state: jnp.ndarray, outputs: dict) -> jnp.ndarray:
        deriv = super().__call__(state, outputs)
        weight = self.condition(outputs)
        return deriv * weight
