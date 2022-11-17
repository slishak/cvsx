from time import perf_counter

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx.unit_conversions import convert


@jax.jit
def main():

    rtol = 1e-3
    atol = 1e-6

    cvs = models.SmoothInertialSmithCVS(
        parameter_source="revie",
        cd=drv.GaussianCardiacDriver(),
    )

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
    ode_solver = diffrax.Kvaerno5()
    term = diffrax.ODETerm(cvs)
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmax=1e-2,
    )

    # out_dbg = cvs(jnp.array(0.0), init_states, (nl_solver,))

    res = diffrax.diffeqsolve(
        term,
        ode_solver,
        0.0,
        10.0,
        None,
        init_states,
        args=(nl_solver,),
        stepsize_controller=stepsize_controller,
        max_steps=16**4,
        saveat=diffrax.SaveAt(steps=True),
    )

    out = cvs(res.ts, res.ys, (nl_solver,), True)

    return res, out


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        print("Compile")
        t0 = perf_counter()
        main()
        t1 = perf_counter()
        print(f"Compiled in {t1-t0:6f}s. Start timing")
        res, out = main()
        t2 = perf_counter()
        print(f'Final time: {t2-t1:.6f}s, {res.stats["num_steps"]} steps')

    fig, ax = plt.subplots(len(res.ys), 2, sharex=True)

    min_q = 0.0

    for i, (key, val) in enumerate(res.ys.items()):
        ax[i, 0].plot(res.ts, val, ".-", label=key)
        ax[i, 1].plot(res.ts, out[f"d{key}_dt"], ".-", label=key)
        ax[i, 0].set_ylabel(key)
        ax[i, 1].set_ylabel(f"d{key}_dt")

    plt.figure()
    plt.hist(jnp.log(jnp.diff(res.ts[~jnp.isinf(res.ts)])))
    plt.xlabel("Log timestep")
    plt.ylabel("Count")

    plt.show()
