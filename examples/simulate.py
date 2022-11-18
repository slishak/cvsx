from time import perf_counter
from functools import partial

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx.unit_conversions import convert


@partial(jax.jit, static_argnums=0)
def main(dynamic=False):

    rtol = 1e-3
    atol = 1e-6

    # f_hr = lambda t: 80 + 20 * jnp.tanh(0.3 * (t - 20))
    if dynamic:
        f_hr = lambda t: jnp.full_like(t, fill_value=60.0)
    else:
        f_hr = 60.0

    cvs = models.SmoothInertialSmithCVS(
        parameter_source="revie",
        cd=drv.SimpleCardiacDriver(hr=f_hr),
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
    if dynamic:
        init_states["s"] = 0.0

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
        40.0,
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
        main(dynamic=False)
        main(dynamic=True)
        t1 = perf_counter()
        print(f"Compiled in {t1-t0:6f}s. Start timing")

        for i in range(4):
            ta = perf_counter()
            res1, out1 = main(dynamic=False)
            tb = perf_counter()
            print(f'{tb-ta:.6f}s, {res1.stats["num_steps"]} steps')

        for i in range(4):
            ta = perf_counter()
            res2, out2 = main(dynamic=True)
            tb = perf_counter()
            print(f'Dynamic HR: {tb-ta:.6f}s, {res2.stats["num_steps"]} steps')

    all_states = res1.ys.keys() | res2.ys.keys()
    fig, ax = plt.subplots(len(all_states), 2, sharex=True)

    min_q = 0.0

    for i, key in enumerate(all_states):
        try:
            ax[i, 0].plot(res1.ts, res1.ys[key], ".-", label=key, markersize=1)
        except KeyError:
            pass
        else:
            ax[i, 1].plot(res1.ts, out1[f"d{key}_dt"], ".-", label=key, markersize=1)
        try:
            ax[i, 0].plot(res2.ts, res2.ys[key], ".-", label=key, markersize=1)
        except KeyError:
            pass
        else:
            ax[i, 1].plot(res2.ts, out2[f"d{key}_dt"], ".-", label=key, markersize=1)
        ax[i, 0].set_ylabel(key)
        ax[i, 1].set_ylabel(f"d{key}_dt")

    plt.figure()
    plt.plot(res1.ts, out1["e_t"], label="Static")
    plt.plot(res2.ts, out2["e_t"], label="Dynamic")

    # plt.figure()
    # plt.hist(jnp.log(jnp.diff(res.ts[~jnp.isinf(res.ts)])))
    # plt.xlabel("Log timestep")
    # plt.ylabel("Count")

    plt.show()
