import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from cvsx import cardiac_drivers as drv

hr = 40
cd = drv.LearnedHR(guess_hr=hr)
t = jnp.linspace(-2, 100, 10000)
plt.plot(t, cd(t))
plt.plot(cd.t_sample(), cd(cd.t_sample()), "x")
plt.show()
