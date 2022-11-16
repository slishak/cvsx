import jax
import diffrax

from cvsx import models

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@jax.jit
def main():

    cvs = models.InertialSmithCVS(**inertial_params)

    init_states = {
        "q_mt": 245.5813,
        "v_lv": 94.6812,
        "q_av": 0.0,
        "v_ao": 133.3381,
        "v_vc": 329.7803,
        "q_tc": 190.0661,
        "v_rv": 90.7302,
        "q_pv": 0.0,
        "v_pa": 43.0123,
        "v_pu": 808.4579,
    }

    solver = diffrax.Tsit5()  # Slightly more efficient than Dopri5
    term = diffrax.ODETerm(cvs)
    stepsize_controller = diffrax.PIDController(
        rtol=1e-4,
        atol=1e-7,
        dtmax=1e-2,
    )

    res = diffrax.diffeqsolve(
        term,
        solver,
        0.0,
        15.0,
        None,
        init_states,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
        saveat=diffrax.SaveAt(steps=True),
    )

    return res


if __name__ == "__main__":
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
