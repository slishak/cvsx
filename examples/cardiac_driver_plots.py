import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cvsx import cardiac_drivers as drv

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    t = jnp.linspace(0, 1.5, 1000)
    cd_simple = drv.SimpleCardiacDriver()
    cd_gauss = drv.GaussianCardiacDriver()

    e_simple = cd_simple(t)
    e_gauss = cd_gauss(t)

    plt.plot(t, e_simple, ".-", label="Simple")
    plt.plot(t, e_gauss, ".-", label="Gaussian")
    plt.legend(loc="upper right")
    plt.show()
