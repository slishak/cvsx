import jax.numpy as jnp
import jax
from plotly import graph_objects as go

jax.config.update("jax_enable_x64", True)

from cvsx import components as c, parameters as p
from cvsx.unit_conversions import convert


if __name__ == "__main__":
    valve = c.InertialValve(**p.revie_2012["mt"])
    smooth_valve = c.SmoothInertialValve(**p.revie_2012["mt"])

    x = jnp.linspace(-10, 30, 500)
    # x = x.at[-1].set(350.0)
    y = jnp.linspace(-5, 5, 500)
    # y = y.at[1].set(20.0)

    xx, yy = jnp.meshgrid(x, y)

    q_flow = convert(xx, "ml/s")
    dp = convert(yy, "mmHg")

    deriv = valve.flow_rate_deriv(dp, dp * 0, q_flow)
    deriv2 = smooth_valve.flow_rate_deriv(dp, dp * 0, q_flow)

    fig = go.Figure()
    fig.add_surface(x=q_flow, y=dp, z=deriv, opacity=0.6, colorscale="Blues")
    fig.add_surface(x=q_flow, y=dp, z=deriv2, opacity=0.6, colorscale="Oranges")
    fig.add_scatter3d(
        x=[0, 0], y=[min(y), 0], z=[0, 0], mode="lines", line_color="red", line_width=6
    )
    fig.add_scatter3d(
        x=[min(x), 0], y=[0, 0], z=[0, 0], mode="lines", line_color="blue", line_width=6
    )
    fig.update_layout(
        scene={
            "xaxis_title": "q (ml/s)",
            "yaxis_title": "dp (mmHg)",
            "zaxis_title": "dq_dt (ml/s^2)",
        }
    )
    fig.write_html("valve_law.html")