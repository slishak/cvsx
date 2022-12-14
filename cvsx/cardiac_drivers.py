from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


class CardiacDriverBase(ABC, eqx.Module):
    dynamic: bool = False


class FixedCardiacDriver(CardiacDriverBase):
    hr: Union[float, Callable]

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


class SimpleCardiacDriver(FixedCardiacDriver):
    b: jnp.ndarray = jnp.array(80.0)

    def e(self, t: jnp.ndarray) -> jnp.ndarray:
        if self.dynamic:
            hr = self.hr(t)
        else:
            hr = self.hr
        return jnp.exp(-self.b * (t - 30 / hr) ** 2)


class GaussianCardiacDriver(FixedCardiacDriver):
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray

    def e(self, t: jnp.ndarray) -> jnp.ndarray:

        t_1d = jnp.atleast_1d(t)
        f = lambda t_i: jnp.sum(self.a * jnp.exp(-self.b * (t_i - self.c) ** 2))
        f_v = jax.vmap(f, 0, 0)
        e_t = f_v(t_1d)

        return jnp.reshape(e_t, t.shape)


class LearnedHR(SimpleCardiacDriver):
    beat_array: jnp.ndarray
    warp_array: jnp.ndarray
    inds: jnp.ndarray
    e_sample: jnp.ndarray
    offset: jnp.ndarray
    min_interval: float
    max_interval: float
    n_beats: int

    def __init__(
        self,
        n_beats: int = 100,
        guess_hr: float = 60.0,
        min_interval: float = 60 / 200,
        max_interval: float = 60 / 20,
        e_sample: jnp.ndarray = jnp.array([0.05, 0.8]),
    ):
        super().__init__(hr=60, b=80)

        self.n_beats = n_beats
        self.dynamic = False
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.offset = jnp.array(-max_interval)
        self.e_sample = e_sample

        e_inv = self.e_inv(e_sample)
        s_sample = jnp.hstack(
            [
                0.0,
                e_inv,
                0.5,
                1.0 - e_inv[::-1],
                1.0,
            ]
        )
        s_sample_full = jnp.arange(n_beats)[:, None] + s_sample[:-1]
        s_sample_flat = s_sample_full.ravel()

        ds_sample = jnp.diff(s_sample)
        log_sample = jnp.log(ds_sample)
        self.warp_array = jnp.tile(log_sample, [n_beats, 1])

        self.inds = s_sample_flat
        self.beat_array = self.normalise(jnp.full(n_beats, 60 / guess_hr))

    def t_sample(self):
        warp_array = jnp.hstack(
            (jnp.zeros((self.n_beats, 1)), jnp.cumsum(jax.nn.softmax(self.warp_array), 1)[:, :-1])
        )

        beat_lengths = (
            jax.nn.sigmoid(self.beat_array) * (self.max_interval - self.min_interval)
            + self.min_interval
        )

        t_warped = warp_array.T * beat_lengths + jnp.append(0.0, jnp.cumsum(beat_lengths))[:-1]
        t_flat = t_warped.T.ravel() + self.offset

        return t_flat

    def normalise(self, interval):
        return jsp.logit((interval - self.min_interval) / (self.max_interval - self.min_interval))

    def f_interp(self, t):
        return jnp.interp(t, self.t_sample(), self.inds)

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

    def e_inv(self, e_t):
        return 0.5 - jnp.sqrt(-jnp.log(e_t) / self.b)
