import jax.numpy as jnp
import jax
from plotly import graph_objects as go

jax.config.update("jax_enable_x64", True)

from cvsx import components as c, parameters as p
from cvsx.unit_conversions import convert


if __name__ == "__main__":
    valves = {
        "standard": c.Valve(
            **p.revie_2012["mt"],
            inertial=True,
        ),
        "regurgitating": c.TwoWayValve(
            **p.revie_2012["mt"],
            inertial=True,
            method="regurgitating",
        ),
        "restoring": c.TwoWayValve(
            **p.revie_2012["mt"],
            inertial=True,
            method="restoring",
        ),
        "restoring_continuous": c.TwoWayValve(
            **p.revie_2012["mt"],
            inertial=True,
            method="restoring_continuous",
        ),
        "smooth": c.SmoothValve(
            **p.revie_2012["mt"],
            inertial=True,
        ),
    }

    x = jnp.linspace(-10, 40, 500)
    # x = x.at[-1].set(350.0)
    y = jnp.linspace(-5, 5, 500)
    # y = y.at[1].set(20.0)

    xx, yy = jnp.meshgrid(x, y)

    q_flow = convert(xx, "ml/s")
    dp = convert(yy, "mmHg")

    fig_2d = go.Figure()
    fig_2d.update_xaxes(title_text="q (l/s)")
    fig_2d.update_yaxes(title_text="dq_dt (l/s^2)")

    fig_3d = go.Figure()
    fig_3d.add_scatter3d(
        x=[0, 0],
        y=[dp.min(), 0],
        z=[0, 0],
        mode="lines",
        line_color="red",
        line_width=6,
        showlegend=False,
    )
    fig_3d.add_scatter3d(
        x=[q_flow.min(), 0],
        y=[0, 0],
        z=[0, 0],
        mode="lines",
        line_color="blue",
        line_width=6,
        showlegend=False,
    )
    fig_3d.update_layout(
        scene={
            "xaxis_title": "q (l/s)",
            "yaxis_title": "dp (mmHg)",
            "zaxis_title": "dq_dt (l/s^2)",
        }
    )

    colorscales = ["Blues", "Reds", "Greens", "Purples", "Oranges"]
    dash = {
        "standard": "solid",
        "regurgitating": "solid",
        "restoring": "solid",
        "restoring_continuous": "solid",
        "smooth": "solid",
    }

    for i, (name, valve) in enumerate(valves.items()):
        deriv = valve.flow_rate_deriv(0, dp, dp * 0, q_flow)
        fig_2d.add_scatter(
            x=q_flow[80, :],
            y=deriv[80, :],
            name=name,
            line_dash=dash[name],
            line_width=4 if name == "standard" else 2,
        )
        fig_3d.add_surface(
            x=q_flow,
            y=dp,
            z=deriv,
            opacity=0.6,
            name=name,
            showscale=False,
            colorscale=colorscales[i],
        )

    fig_2d.write_html("valve_law_2d.html")
    fig_3d.write_html("valve_law_3d.html")
