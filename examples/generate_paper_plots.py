from dataclasses import dataclass
from time import perf_counter
from itertools import repeat
from typing import Union

import jax
import jax.numpy as jnp
import plotly.io as pio

jax.config.update("jax_enable_x64", True)
pio.templates.default = "plotly_white"
# https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721
pio.kaleido.scope.mathjax = None

import plots
import simulate as s


@dataclass
class PlotDefinition:
    name: str
    runs: dict
    plots: dict
    n_repeats: int = 1
    display_mode: str = "dash"
    showlegend: Union[bool, str] = "all"

    def compile(self):
        for name, kwargs in self.runs.items():
            print(f"Compile: {self.name}/{name}")
            t0 = perf_counter()
            s.main(**kwargs)
            t1 = perf_counter()
            print(f"Compiled in {t1-t0:6f}s")

    def simulate(self):
        results = {}
        for name, kwargs in self.runs.items():
            for i in range(self.n_repeats):
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
                ) = s.main(**kwargs)
                tb = perf_counter()
                print(
                    f"{name}: {tb-ta:.6f}s, "
                    f'{res.stats["num_steps"]} steps, '
                    f"{res.t1:.1f}s of simulation"
                )

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

        return results

    def plot(self):
        self.compile()
        results = self.simulate()

        for file, func in self.plots.items():
            if self.display_mode == "dash":
                dash = iter(["solid", "dash", "dot"])
                colour = repeat(None)
            elif self.display_mode == "colour":
                dash = repeat(None)
                colour = iter(plots.qualitative.Plotly)
            else:
                raise NotImplementedError(self.display_mode)
            fig = None
            for i, (
                name,
                (
                    ts,
                    ys,
                    deriv,
                    out,
                    t_dense,
                    y_dense,
                    deriv_dense,
                    deriv_dense_approx,
                    out_dense,
                ),
            ) in enumerate(results.items()):
                fig = func(
                    t_dense,
                    out_dense
                    | y_dense
                    | {f"d{key}_dt": val for key, val in deriv_dense_approx.items()},
                    fig,
                    colour=next(colour),
                    dash=next(dash),
                    group=name if len(results) > 1 else None,
                    mode="lines",
                    showlegend=self.showlegend,
                )
            fig.write_html(f"plots/{self.name}-{file}.html", include_mathjax="cdn")
            fig.write_image(f"plots/{self.name}-{file}.pdf", width=1200, height=600)


if __name__ == "__main__":

    definitions = [
        PlotDefinition(
            "inertial-compare",
            s.inertial_comparison(
                # dtmax=1.0,
                t1=2.0,
                rtol=1e-4,
                atol=1e-7,
            ),
            {
                "lv": plots.plot_lv_pressures,
                "rv": plots.plot_rv_pressures,
                "vent": plots.plot_vent_interaction,
            },
        ),
        PlotDefinition(
            "v_spt-method-compare",
            s.ventricular_interaction_comparison(
                # dtmax=1e-2,
                t1=20.0,
                rtol=1e-4,
                atol=1e-7,
                t_stabilise=0.0,
                jallon=True,
                inertial=False,
                beta=0.0,
                hb=1.0,
                max_steps=16**4,
            ),
            {
                # "lv": plots.plot_lv_pressures,
                # "rv": plots.plot_rv_pressures,
                "spt_resp": plots.plot_spt_resp,
                "vent": plots.plot_vent_interaction,
            },
            display_mode="colour",
            showlegend=True,
        ),
        PlotDefinition(
            "jallon",
            {
                "Jallon": {
                    "jallon": True,
                    "inertial": False,
                    "v_spt_method": "jallon",
                    "beta": 0.0,
                    "hb": 1.0,
                    # "dtmax": 1e-2,
                    "t1": 100.0,
                    "rtol": 1e-4,
                    "atol": 1e-7,
                    "t_stabilise": 20.0,
                    "max_steps": 16**4,
                }
            },
            {
                "lv": plots.plot_lv_pressures,
                "rv": plots.plot_rv_pressures,
                "vent": plots.plot_vent_interaction,
            },
        ),
        PlotDefinition(
            "jallon-stab",
            {
                "Stabilised": {
                    "jallon": True,
                    "inertial": False,
                    "v_spt_method": "jallon",
                    "beta": 0.1,
                    "hb": 0.0,
                    # "dtmax": 1e-2,
                    "t1": 100.0,
                    "rtol": 1e-4,
                    "atol": 1e-7,
                    "t_stabilise": 20.0,
                    "max_steps": 16**4,
                }
            },
            {
                "lv": plots.plot_lv_pressures,
                "rv": plots.plot_rv_pressures,
                "vent": plots.plot_vent_interaction,
            },
        ),
        PlotDefinition(
            "var-hr",
            {
                "Variable HR": {
                    "inertial": False,
                    "dynamic_hr": True,
                    "t_stabilise": 10.0,
                    "t1": 30.0,
                    "max_steps": 16**4,
                    "rtol": 1e-4,
                    "atol": 1e-7,
                    # "dtmax": 1e-2,
                }
            },
            {
                "lv": plots.plot_lv_pressures,
                "rv": plots.plot_rv_pressures,
                "vent": plots.plot_vent_interaction,
            },
        ),
    ]

    with jax.default_device(jax.devices("cpu")[0]):
        # Generate plots
        for definition in definitions:
            definition.plot()

        # Test timing
        for definition in definitions:
            # import pprint

            for name, kwargs in definition.runs.items():
                kwargs["t1"] = 60.0
                kwargs["max_steps"] = 16**4
                kwargs["t_stabilise"] = 60.0
                # print(name)
                # pprint.pprint(kwargs)
            definition.n_repeats = 10
            definition.compile()
            definition.simulate()
