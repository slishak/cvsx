import time
import functools as ft

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wfdb
import diffrax
import equinox as eqx
import optax

jax.config.update("jax_enable_x64", True)

from cvsx import models
from cvsx import cardiac_drivers as drv
from cvsx.unit_conversions import convert


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

    # print("\n".join(header.comments))

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
            1e-2,
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
            # ),
            max_steps=16**3,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return res.ys

    def outputs(self, ts, ys, atol=1e-6, rtol=1e-3):
        out = self.model(
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
        return out


def main():
    data = get_mghdb_data()
    ecg_data = jnp.vstack((data["ECG lead I"], data["ECG lead II"], data["ECG lead V"])).T
    # hr = drv.ECG_HR(data["t"], ecg_data, key=jax.random.PRNGKey(20))
    cd = drv.LearnedHR(
        parameter_source="smith",
        guess_hr=42.0
        # hr=hr,
    )
    cvs = models.SmithCVS(
        # parameter_source="revie",
        cd=cd,
    )
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

    ys = model(data["t"])
    out = model.outputs(data["t"], ys)

    fig, ax = plt.subplots(6, 2, sharex=True)
    # ax[0].plot(data["t"], data["ECG lead I"], "k")
    # ax[1].plot(data["t"], data["ECG lead II"], "k")
    # ax[2].plot(data["t"], data["ECG lead V"], "k")

    ax[0, 0].plot(data["t"], data["ART"], "k")
    ax[1, 0].plot(data["t"], data["CVP"], "k")
    ax[2, 0].plot(data["t"], data["PAP"], "k")
    ax[5, 0].plot(data["t"], data["ECG lead I"], "k")
    ax[5, 0].plot(data["t"], data["ECG lead II"], ":k")
    ax[5, 0].plot(data["t"], data["ECG lead V"], "--k")

    ax[0, 0].plot(data["t"], out["p_ao"], "b")
    ax[1, 0].plot(data["t"], out["p_vc"], "b")
    ax[2, 0].plot(data["t"], out["p_pa"], "b")
    ax[3, 0].plot(data["t"], out["e_t"], "b")

    ax[0, 1].plot(data["t"], ys["v_lv"], "b")
    ax[1, 1].plot(data["t"], ys["v_lv"], "b")
    ax[2, 1].plot(data["t"], ys["v_lv"], "b")
    ax[3, 1].plot(data["t"], ys["v_lv"], "b")
    ax[4, 1].plot(data["t"], ys["v_lv"], "b")
    ax[5, 1].plot(data["t"], ys["v_lv"], "b")

    def parameter_filter(tree):
        nodes = [
            tree.y0["v_ao"],
            tree.y0["v_lv"],
            tree.y0["v_pa"],
            tree.y0["v_pu"],
            tree.y0["v_rv"],
            tree.y0["v_vc"],
            # tree.y0["s"],
            # tree.model.vc.e,
            tree.model.pa.e,
            # tree.model.pu.e,
            tree.model.ao.e,
            tree.model.lvf.e,
            tree.model.rvf.e,
            tree.model.lvf.p_0,
            tree.model.rvf.p_0,
            # tree.model.tc.r,
            # tree.model.pv.r,
            tree.model.pul.r,
            # tree.model.mt.r,
            # tree.model.av.r,
            tree.model.sys.r,
            tree.model.cd.array,
            # tree.model.cd.b,
        ]
        # for layer in tree.model.cd.hr.mlp.layers:
        #     nodes.extend([layer.weight, layer.bias])
        return nodes

    n_params = len(parameter_filter(model))

    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        parameter_filter,
        filter_spec,
        replace=[True] * n_params,
    )

    tree = eqx.filter(model, eqx.is_inexact_array)

    @ft.partial(eqx.filter_value_and_grad, arg=filter_spec)
    def grad_loss(model, obs):
        # ys = jax.vmap(model)(data["t"], y0)
        # p_ao, p_vc, p_pa = jax.vmap(model.outputs)(data["t"], ys)
        ys = model(obs["t"])
        out = model.outputs(obs["t"], ys)
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

    optim = optax.adabelief(1e-2)
    opt_state = optim.init(tree)
    original_model = model
    for i in range(200):
        start = time.perf_counter()
        loss, model, opt_state, grad = make_step(model, data, opt_state)
        end = time.perf_counter()
        print(f"Step {i}: loss {loss}, time {end - start}")
        if i % 10 == 0:
            ys = model(data["t"])
            out = model.outputs(data["t"], ys)
            ax[0, 0].plot(data["t"], out["p_ao"], "grey", alpha=0.5)
            ax[1, 0].plot(data["t"], out["p_vc"], "grey", alpha=0.5)
            ax[2, 0].plot(data["t"], out["p_pa"], "grey", alpha=0.5)
            ax[3, 0].plot(data["t"], out["e_t"], "grey", alpha=0.5)

            ax[0, 1].plot(data["t"], ys["v_lv"], "grey", alpha=0.5)
            ax[1, 1].plot(data["t"], ys["v_lv"], "grey", alpha=0.5)
            ax[2, 1].plot(data["t"], ys["v_lv"], "grey", alpha=0.5)
            ax[3, 1].plot(data["t"], ys["v_lv"], "grey", alpha=0.5)
            ax[4, 1].plot(data["t"], ys["v_lv"], "grey", alpha=0.5)
            ax[5, 1].plot(data["t"], ys["v_lv"], "grey", alpha=0.5)
        if jnp.isnan(loss).any():
            break

    ys = model(data["t"])
    out = model.outputs(data["t"], ys)
    ax[0, 0].plot(data["t"], out["p_ao"], "r")
    ax[1, 0].plot(data["t"], out["p_vc"], "r")
    ax[2, 0].plot(data["t"], out["p_pa"], "r")
    ax[3, 0].plot(data["t"], out["e_t"], "r")

    ax[0, 1].plot(data["t"], ys["v_lv"], "r")
    ax[1, 1].plot(data["t"], ys["v_lv"], "r")
    ax[2, 1].plot(data["t"], ys["v_lv"], "r")
    ax[3, 1].plot(data["t"], ys["v_lv"], "r")
    ax[4, 1].plot(data["t"], ys["v_lv"], "r")
    ax[5, 1].plot(data["t"], ys["v_lv"], "r")
    # ax[7].plot(data["t"], out["ds_dt"], "r")
    plt.show(block=True)


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
