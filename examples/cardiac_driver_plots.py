import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cvsx import cardiac_drivers as drv

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    t = jnp.linspace(0, 3, 3000)
    # cd_simple = drv.SimpleCardiacDriver(hr=120)
    lhr = drv.LearnedHRWarped(
        # n_beats=10,
        guess_hr=120,
    )

    # e_simple = cd_simple(t)
    # e_l = lhr(t)
    # plt.plot(t, e_l, ".-", label="Unwarped")
    object.__setattr__(lhr, "warp_array", lhr.warp_array.at[:, 0].set(-2))
    object.__setattr__(lhr, "warp_array", lhr.warp_array.at[:, 1].set(-2))
    object.__setattr__(lhr, "warp_array", lhr.warp_array.at[:, 2].set(-2))
    object.__setattr__(lhr, "warp_array", lhr.warp_array.at[:, 3].set(-3))
    object.__setattr__(lhr, "warp_array", lhr.warp_array.at[:, 4].set(-2))
    object.__setattr__(lhr, "warp_array", lhr.warp_array.at[:, 5].set(-2))
    object.__setattr__(lhr, "beat_array", lhr.beat_array.at[6].set(-1))
    object.__setattr__(lhr, "beat_array", lhr.beat_array.at[7].set(-3))
    object.__setattr__(lhr, "beat_array", lhr.beat_array.at[8].set(-4))
    object.__setattr__(lhr, "beat_array", lhr.beat_array.at[9].set(-6))
    e_l = lhr(t)
    plt.plot(t, e_l, ".-", label="Warped")

    # plt.plot(t, e_simple, ".-", label="Simple")
    # plt.plot(t, e_l, ".-", label="Warped")
    plt.legend(loc="upper right")
    plt.show()
