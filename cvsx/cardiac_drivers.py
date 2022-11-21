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


class LearnedHR(SimpleCardiacDriver):
    array: jnp.ndarray
    inds: jnp.ndarray
    min_interval: float
    max_interval: float
    n_subsample: int

    def __init__(
        self,
        parameter_source: str = "smith",
        n_beats: int = 40 * 200 // 60 + 1,
        guess_hr: float = 60.0,
        min_hr: float = 20.0,
        max_hr: float = 200.0,
        n_subsample: int = 2,
    ):
        super().__init__(
            parameter_source,
        )
        self.min_interval = 60 / max_hr
        self.max_interval = 60 / min_hr
        self.array = jnp.full(n_beats * n_subsample, fill_value=self.normalise(60 / guess_hr))
        self.inds = jnp.arange(n_beats * n_subsample) / n_subsample
        self.n_subsample = n_subsample

    def beat_times(self):
        intervals = (
            jax.nn.sigmoid(self.array) * (self.max_interval - self.min_interval) + self.min_interval
        )

        return jnp.cumsum(intervals / self.n_subsample) - self.max_interval

    def normalise(self, interval):
        return jsp.logit((interval - self.min_interval) / (self.max_interval - self.min_interval))

    def f_interp(self, t):
        return jnp.interp(t, self.beat_times(), self.inds)

    def __call__(self, t: jnp.ndarray, s: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        interp_vmap = jax.vmap(
            self.f_interp,
            0,
            -1,
        )
        t_1d = jnp.atleast_1d(t)
        s = interp_vmap(t_1d)
        s = jnp.reshape(s, t.shape)
        s_wrapped = jnp.remainder(s, 1)
        e_s = self.e(s_wrapped)
        return e_s
