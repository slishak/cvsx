from time import perf_counter

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

from cvsx import models
from cvsx.unit_conversions import convert

jax.config.update("jax_enable_x64", True)


@jax.jit
def main():

    rtol = 1e-4
    atol = 1e-7

    cvs = models.InertialSmithCVS(parameter_source="revie")

    init_states = {
        "v_lv": convert(94.6812, "ml"),
        "v_ao": convert(133.3381, "ml"),
        "v_vc": convert(329.7803, "ml"),
        "v_rv": convert(90.7302, "ml"),
        "v_pa": convert(43.0123, "ml"),
        "v_pu": convert(808.4579, "ml"),
        "q_mt": convert(245.5813, "ml/s"),
        "q_av": convert(0.0, "ml/s"),
        "q_tc": convert(190.0661, "ml/s"),
        "q_pv": convert(0.0, "ml/s"),
    }
    nl_solver = diffrax.NewtonNonlinearSolver(
        rtol=rtol,
        atol=atol,
    )
    ode_solver = diffrax.Tsit5()  # Slightly more efficient than Dopri5
    term = diffrax.ODETerm(cvs)
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmax=1e-2,
    )

    res = diffrax.diffeqsolve(
        term,
        ode_solver,
        0.0,
        15.0,
        None,
        init_states,
        args=(nl_solver,),
        stepsize_controller=stepsize_controller,
        max_steps=16**4,
        saveat=diffrax.SaveAt(steps=True),
    )

    return res


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        print("Compile")
        t0 = perf_counter()
        main()
        t1 = perf_counter()
        print(f"Compiled in {t1-t0:6f}s. Start timing")
        res = main()
        t2 = perf_counter()
        print(f'Final time: {t2-t1:.6f}s, {res.stats["num_steps"]} steps')

    fig, ax = plt.subplots(3, 1, sharex=True)

    min_q = 0.0

    for key, val in res.ys.items():
        i = 0 if key[0] == "v" else 1
        ax[i].plot(res.ts, val, label=key)
        if key[0] == "q":
            min_q = min(min_q, min(val[~jnp.isinf(val)]))
            ax[2].plot(res.ts, val, label=key)

    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Volume")
    ax[1].set_ylabel("Flow rate")
    ax[2].set_ylabel("Flow rate (zoomed to 0)")
    ax[2].set_ylim([min_q * 1.5, -min_q * 1.5])

    plt.figure()
    plt.hist(jnp.log(jnp.diff(res.ts[~jnp.isinf(res.ts)])))
    plt.xlabel("Log timestep")
    plt.ylabel("Count")

    plt.show()
