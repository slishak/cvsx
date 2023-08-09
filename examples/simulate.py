from time import perf_counter
from functools import partial

import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx import respiratory as resp
from cvsx import components as c, parameters as p
from cvsx.unit_conversions import convert
import plots


@partial(jax.jit, static_argnums=list(range(12)))
def main(
    dynamic_hr=False,
    inertial=True,
    jallon=False,
    valve_type="smith",
    parameter_source="revie",
    v_spt_method="solver",
    ode_solver=diffrax.Tsit5(),
    max_steps=16**3,
    fixed_step=False,
    dt_dense=1e-3,
    t1=20.0,
    t_stabilise=0.0,
    dt_fixed=1e-3,
    beta=0.1,
    hb=1.0,
    mu=1.0,
    rtol=1e-3,
    atol=1e-6,
    dtmax=1e-2,
    p_pl=convert(-4.0, "mmHg"),
    r_reverse=convert(1.0, "mmHg/ml"),
):
    if dynamic_hr:
        f_hr = lambda t: 80 + 20 * jnp.tanh(0.3 * (t - 20))
        # f_hr = lambda t: jnp.full_like(t, fill_value=60.0)
        cd = drv.SimpleCardiacDriver(hr=f_hr)
    else:
        cd = None

    model = models.SmithCVS
    if jallon:
        parameter_source = "jallon_hr_only"

    match valve_type:
        case "smith":
            valve_class = c.Valve
        case "regurgitating":
            valve_class = {
                "mt": partial(
                    c.TwoWayValve,
                    r_reverse=r_reverse,
                ),
                "tc": c.Valve,
                "av": c.Valve,
                "pv": c.Valve,
            }
        case "smooth":
            if not inertial:
                raise RuntimeError("Smooth valves must use inertial model")
            valve_class = partial(
                c.SmoothValve,
                r_reverse=r_reverse,
            )
        case "restoring" | "restoring_continuous":
            if not inertial:
                raise RuntimeError("Restoring valves must use inertial model")
            valve_class = partial(
                c.TwoWayValve,
                r_reverse=r_reverse,
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
    params["p_pl"] = p_pl

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

    term = diffrax.ODETerm(cvs)

    if fixed_step:
        stepsize_controller = diffrax.ConstantStepSize()
        dt0 = dt_fixed
        max_steps = int(t1 / dt0)
    else:
        stepsize_controller = diffrax.PIDController(
            rtol=rtol,
            atol=atol,
            dtmax=dtmax,
            pcoeff=0.4,
            icoeff=0.3,
            dcoeff=0.0,
        )
        dt0 = None

    res = diffrax.diffeqsolve(
        term,
        ode_solver,
        0.0,
        t1,
        dt0,
        init_states,
        args=(nl_solver,),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(steps=True, dense=True),
    )

    deriv, out = cvs(res.ts, res.ys, (nl_solver,), return_outputs=True)

    t_dense = jnp.linspace(t_stabilise, t1, int((t1 - t_stabilise) / dt_dense) + 1)
    y_dense = jax.vmap(res.evaluate)(t_dense)
    deriv_dense, out_dense = cvs(t_dense, y_dense, (nl_solver,), return_outputs=True)
    deriv_dense_approx = jax.vmap(res.derivative)(t_dense)

    return res, deriv, out, t_dense, y_dense, deriv_dense, deriv_dense_approx, out_dense


def jallon_sweep(
    variable="beta",
    values=(0.0, 0.1, 0.5, 1.0, 1.5, 2.0),
    **kwargs,
):
    runs = {
        f"Jallon ({variable}={val})": kwargs
        | {"jallon": True, "inertial": False, "v_spt_method": "jallon", variable: val}
        for val in values
    }

    return runs


def p_pl_sweep(**kwargs):
    runs = {f"p_pl={val}": kwargs | {"p_pl": val} for val in [-2, -4, -6, -8, -10]}
    return runs


def jallon_comparison(**kwargs):
    jallon_kwargs = {
        "jallon": True,
        "inertial": False,
        "v_spt_method": "jallon",
    }
    runs = {
        "Jallon": kwargs | jallon_kwargs | {"beta": 0.0, "hb": 1.0},
        "Stabilised": kwargs | jallon_kwargs | {"beta": 0.1, "hb": 1.0},
        "Stabilised, HB=0": kwargs | jallon_kwargs | {"beta": 0.1, "hb": 0.0},
    }

    return runs


def inertial_comparison(**kwargs):
    runs = {
        "Non-inertial": kwargs | {"inertial": False},
        "Inertial": kwargs | {"inertial": True},
    }

    return runs


def rtol_sweep(**kwargs):
    runs = {
        "1e-3": kwargs | {"rtol": 1e-3},
        "1e-4": kwargs | {"rtol": 1e-4},
        "1e-5": kwargs | {"rtol": 1e-5},
        "1e-6": kwargs | {"rtol": 1e-6},
    }

    return runs


def atol_sweep(**kwargs):
    runs = {
        "1e-5": kwargs | {"atol": 1e-5},
        "1e-6": kwargs | {"atol": 1e-6},
        "1e-7": kwargs | {"atol": 1e-7},
        "1e-8": kwargs | {"atol": 1e-8},
    }

    return runs


def ventricular_interaction_comparison(**kwargs):
    runs = {
        "Linearised": kwargs | {"v_spt_method": "jallon"},
        "Standard": kwargs | {"v_spt_method": "solver"},
        # "None": kwargs | {"v_spt_method": "off"},
    }

    return runs


def valve_comparison(
    restoring=True,
    restoring_continuous=True,
    smooth=True,
    mitral_regurgitation=True,
    **kwargs,
):
    runs = {"standard": kwargs | {"inertial": True}}

    if restoring:
        runs["restoring"] = kwargs | {"inertial": True, "valve_type": "restoring"}

    if restoring_continuous:
        runs["restoring_continuous"] = kwargs | {
            "inertial": True,
            "valve_type": "restoring_continuous",
        }

    if smooth:
        runs["smooth"] = kwargs | {"inertial": True, "valve_type": "smooth"}

    if mitral_regurgitation:
        runs["mitral_regurgitation"] = kwargs | {"inertial": True, "valve_type": "regurgitating"}

    return runs


if __name__ == "__main__":
    # runs = jallon_sweep(
    #     variable="hb",
    #     values=(0.0, 0.5, 1.0),
    #     t1=60.0,
    #     dtmax=1e-2,
    #     rtol=1e-4,
    #     atol=1e-7,
    #     max_steps=16**5,
    # )
    runs = p_pl_sweep(
        t1=20.0,
        rtol=1e-4,
        atol=1e-7,
        t_stabilise=0.0,
        # jallon=True,
        inertial=False,
        # beta=0.0,
        # hb=1.0,
        max_steps=16**4,
    )
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
                (
                    res,
                    deriv,
                    out,
                    t_dense,
                    y_dense,
                    deriv_dense,
                    deriv_dense_approx,
                    out_dense,
                ) = main(**kwargs)
                tb = perf_counter()
                print(f'{name}: {tb-ta:.6f}s, {res.stats["num_steps"]} steps')

                valid_inds = jnp.isfinite(res.ts)
                ts = res.ts[valid_inds]
                ys = {key: val[valid_inds] for key, val in res.ys.items()}
                deriv = {key: val[valid_inds] for key, val in deriv.items()}
                out = {key: val[valid_inds] for key, val in out.items()}

                results[name] = (
                    ts,
                    ys,
                    deriv,
                    out,
                    t_dense,
                    y_dense,
                    deriv_dense,
                    deriv_dense_approx,
                    out_dense,
                )

    plot_dict = {
        "lv": plots.plot_lv_pressures,
        "rv": plots.plot_rv_pressures,
        "vent": plots.plot_vent_interaction,
        "outputs": plots.plot_outputs,
        "resp": plots.plot_resp,
        "states": plots.plot_states,
    }

    for file, func in plot_dict.items():
        fig = None
        for i, (
            name,
            (ts, ys, deriv, out, t_dense, y_dense, deriv_dense, deriv_dense_approx, out_dense),
        ) in enumerate(results.items()):
            fig = func(
                ts,
                out | ys | {f"d{key}_dt": val for key, val in deriv.items()},
                fig,
                plots.qualitative.Plotly[i],
                group=name,
                mode="markers",
                marker_size=4,
            )
            fig = func(
                t_dense,
                out_dense
                | y_dense
                | {f"d{key}_dt": val for key, val in deriv_dense_approx.items()},
                fig,
                plots.qualitative.Plotly[i],
                group=name,
                mode="lines",
                showlegend=False,
            )
        fig.write_html(f"{file}.html", include_mathjax="cdn")

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
