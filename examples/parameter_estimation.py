import time
import functools as ft
import traceback

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wfdb
import diffrax
import equinox as eqx
import optax
import jaxopt
from plotly.subplots import make_subplots

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx import parameters as p
from cvsx.unit_conversions import convert
import plots


def get_mghdb_data(
    record_name: str = "mgh001",
    record_dir: str = "mghdb/1.0.0/.",
    start_s: float = 720.0,
    end_s: float = 760.0,
):

    header = wfdb.rdheader(
        record_name=record_name,
        pn_dir=record_dir,
    )

    print("\n".join(header.comments))

    fs = header.fs
    sampfrom = int(fs * start_s)
    sampto = int(fs * end_s)

    record = wfdb.rdrecord(
        record_name=record_name,
        pn_dir=record_dir,
        sampfrom=sampfrom,
        sampto=sampto,
    )

    record_data = {key: jnp.array(record.p_signal[:, i]) for i, key in enumerate(record.sig_name)}
    record_data["t"] = jnp.arange(0, (end_s - start_s) * fs) / fs

    # wfdb.plot_wfdb(record)

    return record_data


class TrainableODE(eqx.Module):
    model: models.SmithCVS
    y0: dict

    def __call__(self, ts, atol=1e-6, rtol=1e-3):
        res = diffrax.diffeqsolve(
            diffrax.ODETerm(self.model),
            diffrax.Heun(),
            ts[0],
            ts[-1],
            2e-3,
            self.y0,
            args=[
                diffrax.NewtonNonlinearSolver(
                    rtol=rtol,
                    atol=atol,
                )
            ],
            # stepsize_controller=diffrax.PIDController(
            #     rtol=rtol,
            #     atol=atol,
            #     dtmax=1e-2,
            #     pcoeff=0.4,
            #     icoeff=0.3,
            #     dcoeff=0,
            # ),
            max_steps=int(40 / 2e-3),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return res.ys

    def outputs(self, ts, ys, atol=1e-6, rtol=1e-3):
        derivs, outputs = self.model(
            ts,
            ys,
            [
                diffrax.NewtonNonlinearSolver(
                    rtol=rtol,
                    atol=atol,
                )
            ],
            True,
        )
        return derivs, outputs


def plot_solution(ax, t, ys, model, outputs, color, alpha=1.0, t_ticks=True):
    ax[0, 0].plot(t, outputs["p_ao"], color=color, alpha=alpha)
    ax[1, 0].plot(t, outputs["p_vc"], color=color, alpha=alpha)
    ax[2, 0].plot(t, outputs["p_pa"], color=color, alpha=alpha)
    ax[3, 0].plot(t, outputs["e_t"], color=color, alpha=alpha)
    if t_ticks:
        ax[3, 0].plot(model.model.cd.t_sample(), model.model.cd.inds * 0, color=color, marker="|")

    ax[0, 1].plot(t, ys["v_lv"], color=color, alpha=alpha)
    ax[1, 1].plot(t, ys["v_ao"], color=color, alpha=alpha)
    ax[2, 1].plot(t, ys["v_vc"], color=color, alpha=alpha)
    ax[3, 1].plot(t, ys["v_rv"], color=color, alpha=alpha)
    ax[4, 1].plot(t, ys["v_pa"], color=color, alpha=alpha)
    ax[5, 1].plot(t, ys["v_pu"], color=color, alpha=alpha)
    ax[5, 0].plot(t, sum(v for k, v in ys.items() if k[0] == "v"), color=color, alpha=alpha)


def main(inertial=False):
    data = get_mghdb_data()
    # ecg_data = jnp.vstack((data["ECG lead I"], data["ECG lead II"], data["ECG lead V"])).T
    # hr = drv.ECG_HR(data["t"], ecg_data, key=jax.random.PRNGKey(20))
    cd = drv.LearnedHR(guess_hr=42.0, n_beats=40, e_sample=jnp.array([0.05, 0.5, 0.8, 0.95]))
    params = p.build_parameter_tree(
        "revie",
        inertial,
        cd,
    )
    cvs = models.SmithCVS(**params, v_spt_method="jallon")
    if inertial:
        y0 = {
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
        y0 = {
            "v_lv": jnp.array(convert(137.5, "ml")),
            "v_ao": jnp.array(convert(951.5, "ml")),
            "v_vc": jnp.array(convert(3190.0, "ml")),
            "v_rv": jnp.array(convert(132.0, "ml")),
            "v_pa": jnp.array(convert(187.0, "ml")),
            "v_pu": jnp.array(convert(902.0, "ml")),
            # "s": jnp.array(0.0),
        }

    model = TrainableODE(cvs, y0)

    fig, ax = plt.subplots(6, 2, sharex=True)

    ax[0, 0].plot(data["t"], data["ART"], "k")
    ax[1, 0].plot(data["t"], data["CVP"], "k")
    ax[2, 0].plot(data["t"], data["PAP"], "k")
    # ax[5, 0].plot(data["t"], data["ECG lead I"], "k")
    # ax[5, 0].plot(data["t"], data["ECG lead II"], ":k")
    # ax[5, 0].plot(data["t"], data["ECG lead V"], "--k")

    ys = model(data["t"])
    deriv, outputs = model.outputs(data["t"], ys)
    plot_solution(ax, data["t"], ys, model, outputs, "b")

    def parameter_filter(tree):
        nodes = [
            tree.y0["v_ao"],
            tree.y0["v_lv"],
            tree.y0["v_pa"],
            tree.y0["v_pu"],
            tree.y0["v_rv"],
            tree.y0["v_vc"],
            # tree.y0["s"],
            tree.model.vc._e,
            tree.model.pa._e,
            tree.model.pu._e,
            tree.model.ao._e,
            tree.model.lvf._e,
            tree.model.rvf._e,
            tree.model.spt._e,
            tree.model.lvf._lam,
            tree.model.rvf._lam,
            tree.model.spt._lam,
            tree.model.pcd._lam,
            tree.model.lvf.p_0,
            tree.model.rvf.p_0,
            tree.model.spt.p_0,
            tree.model.pcd.p_0,
            tree.model.lvf.v_0,
            tree.model.rvf.v_0,
            tree.model.spt.v_0,
            tree.model.pcd.v_0,
            tree.model.pul._r,
            tree.model.tc._r,
            tree.model.pv._r,
            tree.model.mt._r,
            tree.model.av._r,
            tree.model.sys._r,
            tree.model.p_pl,
            tree.model.cd.beat_array,
            tree.model.cd.warp_array,
            tree.model.cd.offset,
            # tree.model.cd.b,
        ]
        if inertial:
            nodes.extend(
                [
                    tree.y0["q_mt"],
                    tree.y0["q_tc"],
                    tree.y0["q_av"],
                    tree.y0["q_pv"],
                    tree.model.tc._l,
                    tree.model.pv._l,
                    tree.model.mt._l,
                    tree.model.av._l,
                ]
            )
        return nodes

    n_params = len(parameter_filter(model))

    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        parameter_filter,
        filter_spec,
        replace=[True] * n_params,
    )

    @ft.partial(eqx.filter_value_and_grad, arg=filter_spec)
    def grad_loss(model, obs):
        ys = model(obs["t"])
        deriv, out = model.outputs(obs["t"], ys)
        art_err = out["p_ao"] - obs["ART"]
        cvp_error = out["p_vc"] - obs["CVP"]
        pap_error = out["p_pa"] - obs["PAP"]
        total_error = art_err**2 + cvp_error**2 + pap_error**2
        return jnp.mean(total_error)

    @eqx.filter_jit
    def make_step(model, obs, opt_state):
        loss, grads = grad_loss(model, obs)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads

    # @ft.partial(jax.jit, static_argnums=1)
    def obj_bfgs(tree, static_tree, obs):
        model = eqx.combine(tree, static_tree)
        loss, grads = grad_loss(model, obs)
        jax.debug.print("{loss}", loss=loss)
        return loss, grads

    lr = optax.piecewise_interpolate_schedule(
        interpolate_type="linear",
        init_value=2e-2,
        boundaries_and_scales={
            10: 1,
            25: 0.5,
            # 50: 0.5,
            # 100: 0.5,
            # 150: 0.5,
            # 200: 0.5,
            # 300: 0.5,
        },
    )
    n_epochs_sgd = 300

    optim = optax.adabelief(lr)
    tree, static_tree = eqx.partition(model, filter_spec)
    opt_state = optim.init(tree)
    model_list = []
    for i in range(n_epochs_sgd):
        start = time.perf_counter()
        try:
            loss, new_model, opt_state, grad = make_step(model, data, opt_state)
        except ValueError as ex:
            traceback.print_exc()
            break
        else:
            end = time.perf_counter()
            print(f"Step {i}: loss {loss}, time {end - start}")
            if jnp.isnan(loss).any():
                break
        model_list.append((loss, model))
        model = new_model
        if i % 10 == 0:
            ys = model(data["t"])
            deriv, outputs = model.outputs(data["t"], ys)
            plot_solution(ax, data["t"], ys, model, outputs, "grey", alpha=0.5, t_ticks=False)

    # solver = jaxopt.BFGS(
    #     lambda x: obj_bfgs(x, static_tree, data),
    #     value_and_grad=True,
    #     max_stepsize=1e-4,
    #     min_stepsize=1e-9,
    #     # stepsize=1e-5,
    # )
    # # loss, grad = obj_bfgs(tree, static_tree, data)
    # tree, static_tree = eqx.partition(model, filter_spec)
    # res = solver.run(tree)
    # model = eqx.combine(res.params, static_tree)

    ys = model(data["t"])
    deriv, outputs = model.outputs(data["t"], ys)
    plot_solution(ax, data["t"], ys, model, outputs, "r")
    plots.plot_lv_pressures(data["t"], outputs).write_html("lv.html", include_mathjax="cdn")
    plots.plot_rv_pressures(data["t"], outputs).write_html("rv.html", include_mathjax="cdn")

    fig, ax = plt.subplots(6, 1, sharex=True)
    for key in y0:
        ax[0].plot([model.y0[key] for loss, model in model_list], label=key)
    ax[1].plot([model.model.pa.e for loss, model in model_list], label="e_pa")
    ax[1].plot([model.model.ao.e for loss, model in model_list], label="e_ao")
    ax[1].plot([model.model.lvf.e for loss, model in model_list], label="e_lvf")
    ax[1].plot([model.model.rvf.e for loss, model in model_list], label="e_rvf")
    ax[2].plot([model.model.lvf.p_0 for loss, model in model_list], label="p0_lvf")
    ax[2].plot([model.model.rvf.p_0 for loss, model in model_list], label="p0_rvf")
    ax[3].plot([model.model.pul.r for loss, model in model_list], label="r_pul")
    ax[3].plot([model.model.sys.r for loss, model in model_list], label="r_sys")
    ax[3].plot([model.model.tc.r for loss, model in model_list], label="r_tc")
    ax[3].plot([model.model.mt.r for loss, model in model_list], label="r_mt")
    ax[3].plot([model.model.pv.r for loss, model in model_list], label="r_pv")
    ax[3].plot([model.model.av.r for loss, model in model_list], label="r_av")
    ax[4].plot([loss for loss, model in model_list], label="loss")
    ax[5].plot(jax.vmap(lr)(jnp.arange(n_epochs_sgd)), label="lr")

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    ax[3].legend(loc="upper right")

    plt.show(block=True)


if __name__ == "__main__":
    main()
