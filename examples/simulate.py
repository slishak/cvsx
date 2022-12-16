from time import perf_counter
from functools import partial

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import plotly.graph_objects as go

jax.config.update("jax_enable_x64", True)

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx import respiratory as resp
from cvsx import components as c, parameters as p
from cvsx.unit_conversions import convert
import plots


@partial(jax.jit, static_argnums=[0, 1, 2, 3, 4])
def main(
    dynamic_hr=False,
    inertial=True,
    jallon=False,
    valve_type="smith",
    v_spt_method="solver",
    beta=0.1,
    hb=1.0,
    mu=1.0,
):

    rtol = 1e-3
    atol = 1e-6

    if dynamic_hr:
        f_hr = lambda t: 80 + 20 * jnp.tanh(0.3 * (t - 20))
        # f_hr = lambda t: jnp.full_like(t, fill_value=60.0)
    else:
        f_hr = 60.0

    cd = drv.SimpleCardiacDriver(hr=f_hr)

    model = models.SmithCVS
    if jallon:
        parameter_source = "jallon"
    else:
        parameter_source = "revie"  # "smith"

    match valve_type:
        case "smith":
            valve_class = c.Valve
        case "regurgitating":
            valve_class = {
                "mt": c.TwoWayValve,
                "tc": c.Valve,
                "av": c.Valve,
                "pv": c.Valve,
            }
        case "smooth":
            if not inertial:
                raise RuntimeError("Smooth valves must use inertial model")
            valve_class = c.SmoothValve
        case "restoring" | "restoring_continuous":
            if not inertial:
                raise RuntimeError("Restoring valves must use inertial model")
            valve_class = partial(
                c.TwoWayValve,
                r_reverse=convert(10, "mmHg/ml"),
                method=valve_type,
            )
        case _:
            raise RuntimeError(valve_type)

    params = p.build_parameter_tree(
        parameter_source,
        inertial,
        cd,
        valve_class,
    )

    cvs = model(
        **params,
        p_pl_is_input=jallon,
        v_spt_method=v_spt_method,
    )

    nl_solver = diffrax.NewtonNonlinearSolver(
        rtol=rtol,
        atol=atol,
    )

    if parameter_source == "revie":
        init_states = {
            "v_lv": jnp.array(convert(94.6812, "ml")),
            "v_ao": jnp.array(convert(133.3381, "ml")),
            "v_vc": jnp.array(convert(329.7803, "ml")),
            "v_rv": jnp.array(convert(90.7302, "ml")),
            "v_pa": jnp.array(convert(43.0123, "ml")),
            "v_pu": jnp.array(convert(808.4579, "ml")),
        }
    else:
        init_states = {
            "v_lv": jnp.array(convert(137.5, "ml")),
            "v_ao": jnp.array(convert(951.5, "ml")),
            "v_vc": jnp.array(convert(3190.0, "ml")),
            "v_rv": jnp.array(convert(132.0, "ml")),
            "v_pa": jnp.array(convert(187.0, "ml")),
            "v_pu": jnp.array(convert(902.0, "ml")),
        }

    if dynamic_hr:
        init_states["s"] = jnp.array(0.0)

    if inertial:
        p_v = cvs.pressures_volumes(
            jnp.array(0.0), jnp.array(0.0), init_states, nl_solver, cvs.p_pl
        )
        for name, (vessel, p_upstream, p_downstream) in cvs.connections.items():
            if vessel.inertial:
                init_states[name] = vessel.flow_rate(0.0, p_v[p_upstream], p_v[p_downstream])

    if jallon:
        cvs = models.JallonHeartLungs(
            cvs=cvs,
            resp_sys=resp.PassiveRespiratorySystem(),
            resp_pattern=resp.RespiratoryPatternGenerator(
                beta=beta,
                hb=convert(hb, "1/l"),
                mu=convert(mu, "mmHg"),
            ),
        )
        init_states.update(
            {
                "x": jnp.array(-0.6),
                "y": jnp.array(0.0),
                "p_mus": jnp.array(0.0),
                "v_alv": jnp.array(0.5),
            }
        )

    # out_dbg = cvs(jnp.array(0.0), init_states, (nl_solver,))

    ode_solver = diffrax.Heun()
    term = diffrax.ODETerm(cvs)
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmax=1e-2,
        # pcoeff=0.4,
        # icoeff=0.3,
        # dcoeff=0.0,
    )
    res = diffrax.diffeqsolve(
        term,
        ode_solver,
        0.0,
        2.0,
        1e-3,
        init_states,
        args=(nl_solver,),
        # stepsize_controller=stepsize_controller,
        max_steps=16**5,
        saveat=diffrax.SaveAt(steps=True),
        adjoint=diffrax.NoAdjoint(),
    )

    deriv, out = cvs(res.ts, res.ys, (nl_solver,), True)

    return res, deriv, out


if __name__ == "__main__":

    runs = {
        # f"Jallon (beta={0.1})": {
        #     "jallon": True,
        #     "inertial": False,
        #     "beta": 0.1,
        #     "hb": 0.0,
        #     "mu": 1.0,
        # }
        # for beta in [0.1, 0.5, 1.0, 1.5, 2.0]
        # for mu in [0.0, 0.5, 1.0, 1.5, 2.0]
        # "Jallon inertial": {"jallon": True, "inertial": True, "beta": 0.0},
        # "Non-inertial": {"inertial": False},
        "standard": {"inertial": True},
        "mitral regurgitation": {"inertial": True, "valve_type": "regurgitating"},
        "restoring": {"inertial": True, "valve_type": "restoring"},
        "restoring_continuous": {"inertial": True, "valve_type": "restoring_continuous"},
        "smooth": {"inertial": True, "valve_type": "smooth"},
        # "Regurgitating": {"valve_type": "mitral_regurgitation"},
        # "Regurgitating non-inertial": {"valve_type": "mitral_regurgitation", "inertial": False},
    }
    n_repeats = 1

    with jax.default_device(jax.devices("cpu")[0]):
        for name, kwargs in runs.items():
            print(f"Compile: {name}")
            t0 = perf_counter()
            main(**kwargs)
            t1 = perf_counter()
            print(f"Compiled in {t1-t0:6f}s")

        results = {}
        for name, kwargs in runs.items():
            for i in range(n_repeats):
                ta = perf_counter()
                res, deriv, out = main(**kwargs)
                tb = perf_counter()
                print(f'{name}: {tb-ta:.6f}s, {res.stats["num_steps"]} steps')
                results[name] = (res, deriv, out)

    plot_dict = {
        "lv.html": plots.plot_lv_pressures,
        "rv.html": plots.plot_rv_pressures,
        "vent.html": plots.plot_vent_interaction,
        "outputs.html": plots.plot_outputs,
        # "resp.html": plots.plot_resp,
        "states.html": plots.plot_states,
    }

    for file, func in plot_dict.items():
        fig = None
        for i, (name, (res, deriv, out)) in enumerate(results.items()):
            fig = func(
                res.ts,
                out | res.ys | {f"d{key}_dt": val for key, val in deriv.items()},
                fig,
                plots.qualitative.Plotly[i],
                group=name,
            )
        fig.write_html(file, include_mathjax="cdn")

    # res, deriv, out = results["Restoring"]
    # restoring_valve = c.RestoringInertialValve(**p.revie_2012["mt"])
    # dp = out["p_pu"] - out["p_lv"]
    # q = out["q_mt"]
    # dp_a = jnp.linspace(-1, dp[jnp.isfinite(dp)].max(), 200)
    # q_a = jnp.linspace(-0.1, q[jnp.isfinite(q)].max(), 200)
    # dp_m, q_m = jnp.meshgrid(dp_a, q_a)
    # dq_dt_m = restoring_valve.flow_rate_deriv(0, dp_m, dp_m * 0, q_m)
    # dq_dt = deriv["q_mt"]
    # fig = go.Figure()
    # fig.add_scatter3d(
    #     x=q[dp > -1],
    #     y=dp[dp > -1],
    #     z=dq_dt[dp > -1],
    #     mode="lines+markers",
    #     marker_size=1,
    # )
    # fig.add_surface(
    #     x=q_m,
    #     y=dp_m,
    #     z=dq_dt_m,
    #     colorscale="Reds",
    #     opacity=0.7,
    # )
    # fig.update_layout(
    #     scene={
    #         "xaxis_title": "q (l/s)",
    #         "yaxis_title": "dp (mmHg)",
    #         "zaxis_title": "dq_dt (l/s^2)",
    #     }
    # )
    # fig.write_html("mitral_valve.html")
