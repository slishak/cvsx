from time import perf_counter
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx import respiratory as resp
from cvsx.unit_conversions import convert


@partial(jax.jit, static_argnums=[0, 1, 2])
def main(
    dynamic=False,
    inertial=False,
    jallon=False,
):

    rtol = 1e-4
    atol = 1e-7

    if dynamic:
        f_hr = lambda t: 80 + 20 * jnp.tanh(0.3 * (t - 20))
        # f_hr = lambda t: jnp.full_like(t, fill_value=60.0)
    else:
        f_hr = 60.0

    if jallon:
        model = models.SmithCVS
        parameter_source = "jallon"
    else:
        if inertial:
            model = models.InertialSmithCVS
            parameter_source = "revie"
        else:
            model = models.SmithCVS
            parameter_source = "smith"

    cvs = model(
        parameter_source=parameter_source,
        cd=drv.SimpleCardiacDriver(hr=f_hr),
        p_pl_is_input=jallon,
    )

    if jallon:
        cvs = models.JallonHeartLungs(
            cvs=cvs,
            resp_sys=resp.PassiveRespiratorySystem(),
            resp_pattern=resp.RespiratoryPatternGenerator(),
        )

    if inertial:
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
    else:
        init_states = {
            "v_lv": convert(137.5, "ml"),
            "v_ao": convert(951.5, "ml"),
            "v_vc": convert(3190.0, "ml"),
            "v_rv": convert(132.0, "ml"),
            "v_pa": convert(187.0, "ml"),
            "v_pu": convert(902.0, "ml"),
        }

    if jallon:
        init_states.update(
            {
                "x": -0.6,
                "y": 0.0,
                "p_mus": 0.0,
                "v_alv": 0.5,
            }
        )

    if dynamic:
        init_states["s"] = 0.0

    nl_solver = diffrax.NewtonNonlinearSolver(
        rtol=rtol,
        atol=atol,
    )
    ode_solver = diffrax.Tsit5()
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
        5.0,
        None,
        init_states,
        args=(nl_solver,),
        stepsize_controller=stepsize_controller,
        max_steps=16**4,
        saveat=diffrax.SaveAt(steps=True),
        adjoint=diffrax.NoAdjoint(),
    )

    deriv, out = cvs(res.ts, res.ys, (nl_solver,), True)

    return res, deriv, out


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        print("Compile")
        t0 = perf_counter()
        main(inertial=True)
        t1 = perf_counter()
        print(f"Compiled in {t1-t0:6f}s. Start timing")

        for i in range(4):
            ta = perf_counter()
            res1, deriv1, out1 = main(inertial=True)
            tb = perf_counter()
            print(f'{tb-ta:.6f}s, {res1.stats["num_steps"]} steps')

    # all_states = res1.ys.keys()
    # fig, ax = plt.subplots(len(all_states), 2, sharex=True)
    # for i, key in enumerate(all_states):
    #     try:
    #         ax[i, 0].plot(res1.ts, res1.ys[key], ".-", label=key, markersize=2)
    #     except KeyError:
    #         pass
    #     else:
    #         ax[i, 1].plot(res1.ts, deriv1[key], ".-", label=key, markersize=2)
    #     ax[i, 0].set_ylabel(key)
    #     ax[i, 1].set_ylabel(f"d{key}_dt")

    fig, ax = plt.subplots(6, 1, sharex=True)
    ax[0].plot(res1.ts, out1["p_ao"])
    ax[0].plot(res1.ts, out1["p_lv"], ":")
    ax[0].plot(res1.ts, out1["p_pu"], ":")
    ax[0].set_ylabel("p_ao")
    ax[1].plot(res1.ts, out1["q_av"])
    ax[1].set_ylabel("q_av")
    ax[2].plot(res1.ts, deriv1["q_av"])
    ax[2].set_ylabel("dq_av_dt")

    ax[3].plot(res1.ts, out1["p_vc"])
    ax[3].set_ylabel("p_vc")
    ax[4].plot(res1.ts, out1["p_pa"])
    ax[4].set_ylabel("p_pa")
    ax[5].plot(res1.ts, out1["e_t"])
    ax[5].set_ylabel("e(t)")

    # plt.figure()
    # plt.hist(jnp.log(jnp.diff(res.ts[~jnp.isinf(res.ts)])))
    # plt.xlabel("Log timestep")
    # plt.ylabel("Count")

    plt.show()
