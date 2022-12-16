from plotly.subplots import make_subplots
from plotly.colors import qualitative

from cvsx.unit_conversions import convert


def latex_(s: str):
    return rf"$\Large{{{s}}}$"


def latex(s: str):
    return s.replace("{", "").replace("}", "")


def plot_lv_pressures(t, outputs, fig=None, colour=qualitative.Plotly[0], group=None):

    if fig is None:
        fig = make_subplots(2, 1, shared_xaxes="all")
        fig.update_layout(hovermode="x")
        fig.update_yaxes(row=1, col=1, title_text="Pressure (mmHg)")
        fig.update_yaxes(row=2, col=1, title_text="Flow rates (l/s)")
        fig.update_xaxes(row=2, col=1, title_text="Time (s)")

    if group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour,
            mode="lines",
        )
    fig.add_scatter(
        x=t,
        y=convert(outputs["p_lv"], to="mmHg"),
        name=latex("P_{lv}"),
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
        line_width=1,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["p_ao"], to="mmHg"),
        name=latex("P_{ao}"),
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["p_pu"], to="mmHg"),
        name=latex("P_{pu}"),
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
    )

    fig.add_scatter(
        x=t,
        y=outputs["q_mt"],
        name=latex("Q_{mt}"),
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=outputs["q_av"],
        name=latex("Q_{av}"),
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=outputs["q_sys"],
        name=latex("Q_{sys}"),
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )

    return fig


def plot_rv_pressures(t, outputs, fig=None, colour=qualitative.Plotly[0], group=None):

    if fig is None:
        fig = make_subplots(2, 1, shared_xaxes="all")
        fig.update_layout(hovermode="x")
        fig.update_yaxes(row=1, col=1, title_text="Pressure (mmHg)")
        fig.update_yaxes(row=2, col=1, title_text="Flow rates (l/s)")
        fig.update_xaxes(row=2, col=1, title_text="Time (s)")

    if group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour,
            mode="lines",
        )
    fig.add_scatter(
        x=t,
        y=convert(outputs["p_rv"], to="mmHg"),
        name=latex("P_{rv}"),
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
        line_width=1,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["p_pa"], to="mmHg"),
        name=latex("P_{pa}"),
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["p_vc"], to="mmHg"),
        name=latex("P_{vc}"),
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
    )

    fig.add_scatter(
        x=t,
        y=outputs["q_tc"],
        name=latex("Q_{tc}"),
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=outputs["q_pv"],
        name=latex("Q_{pv}"),
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=outputs["q_pul"],
        name=latex("Q_{pul}"),
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )

    return fig


def plot_vent_interaction(t, outputs, fig=None, colour=qualitative.Plotly[0], group=None):

    specs = [
        [{}, {"rowspan": 3}],
        [{}, None],
        [{}, None],
    ]

    if fig is None:
        fig = make_subplots(3, 2, shared_xaxes="columns", specs=specs)
        fig.update_xaxes(row=3, col=1, title_text="Time (s)")
        fig.update_yaxes(row=1, col=1, title_text="Left ventricle volume (ml)")
        fig.update_yaxes(row=2, col=1, title_text="Right ventricle volume (ml)")
        fig.update_yaxes(row=3, col=1, title_text="Septum volume (ml)")
        fig.update_yaxes(col=2, title_text="Ventricle pressure (mmHg)")
        fig.update_xaxes(col=2, title_text="Ventricle volume (ml)")

    if group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour,
            mode="lines",
        )
    fig.add_scatter(
        x=t,
        y=convert(outputs["v_lv"], to="ml"),
        name="Left",
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["v_rv"], to="ml"),
        name="Right",
        legendgroup=group,
        showlegend=group is None,
        row=2,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["v_spt"], to="ml"),
        name=latex("V_{spt}"),
        legendgroup=group,
        showlegend=group is None,
        row=3,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["v_spt"], to="ml"),
        name=latex("V_{spt}"),
        legendgroup=group,
        showlegend=group is None,
        row=3,
        col=1,
        line_color=colour,
    )
    fig.add_scatter(
        x=convert(outputs["v_lv"], to="ml"),
        y=convert(outputs["p_lv"], to="mmHg"),
        name="Left",
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=2,
        line_color=colour,
    )
    fig.add_scatter(
        x=convert(outputs["v_rv"], to="ml"),
        y=convert(outputs["p_rv"], to="mmHg"),
        name="Right",
        legendgroup=group,
        showlegend=group is None,
        row=1,
        col=2,
        line_color=colour,
    )

    return fig


def plot_outputs(t, outputs, fig=None, colour=qualitative.Plotly[0], group=None):
    if fig is None:
        specs = [
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 2}, None, {}, {}],
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 2}, None, {"colspan": 2}, None],
        ]
        fig = make_subplots(len(specs), 4, specs=specs)
        fig.update_layout(hovermode="x")
        fig.update_xaxes(matches="x1")
        fig.update_xaxes(row=3, col=3, matches=None)
        fig.update_xaxes(row=3, col=4, matches=None)

        fig.update_yaxes(row=1, col=1, title_text="lvf/lv/ao/pu pressures (mmHg)")
        fig.update_yaxes(row=1, col=3, title_text="lvf/lv/ao/pu volumes (ml)")
        fig.update_yaxes(row=2, col=1, title_text="rvf/rv/pa/vc pressures (mmHg)")
        fig.update_yaxes(row=2, col=3, title_text="rvf/rv/pa/vc volumes (ml)")
        fig.update_yaxes(row=3, col=1, title_text="Flow rates (l/s)")
        fig.update_xaxes(row=3, col=3, title_text="v_lv")
        fig.update_yaxes(row=3, col=3, title_text="p_lv")
        fig.update_xaxes(row=3, col=4, title_text="v_rv")
        fig.update_yaxes(row=3, col=4, title_text="p_rv")
        fig.update_yaxes(row=4, col=1, title_text="Pericardium pressures (mmHg)")
        fig.update_yaxes(row=4, col=3, title_text="Pericardium volume (ml)")
        fig.update_yaxes(row=5, col=1, title_text="Cardiac driver")
        fig.update_yaxes(row=5, col=3, title_text="Septum volume (ml)")

    for col in ["p_lvf", "p_lv", "p_ao", "p_pu", "p_aom", "p_aos", "p_aod"]:
        try:
            fig.add_scatter(
                x=t,
                y=convert(outputs[col], to="mmHg"),
                name=col,
                legendgroup=group,
                row=1,
                col=1,
                line_color=colour,
            )
        except KeyError:
            pass

    for col in ["v_lvf", "v_lv", "v_ao", "v_pu"]:
        fig.add_scatter(
            x=t,
            y=convert(outputs[col], "l", "ml"),
            name=col,
            legendgroup=group,
            row=1,
            col=3,
            line_color=colour,
        )

    for col in ["p_rvf", "p_rv", "p_pa", "p_vc", "p_vcm"]:
        try:
            fig.add_scatter(
                x=t,
                y=convert(outputs[col], to="mmHg"),
                name=col,
                legendgroup=group,
                row=2,
                col=1,
                line_color=colour,
            )
        except KeyError:
            pass

    for col in ["v_rvf", "v_rv", "v_pa", "v_vc"]:
        fig.add_scatter(
            x=t,
            y=convert(outputs[col], to="ml"),
            name=col,
            legendgroup=group,
            row=2,
            col=3,
            line_color=colour,
        )

    for col in ["q_mt", "q_av", "q_tc", "q_pv", "q_pul", "q_sys"]:
        fig.add_scatter(
            x=t,
            y=outputs[col],
            name=col,
            legendgroup=group,
            row=3,
            col=1,
            line_color=colour,
        )

    fig.add_scatter(
        x=convert(outputs["v_lv"], to="ml"),
        y=convert(outputs["p_lv"], to="mmHg"),
        name="lv",
        legendgroup=group,
        row=3,
        col=3,
        line_color=colour,
    )
    fig.add_scatter(
        x=convert(outputs["v_rv"], to="ml"),
        y=convert(outputs["p_rv"], to="mmHg"),
        name="rv",
        legendgroup=group,
        row=3,
        col=4,
        line_color=colour,
    )
    for col in ["p_pcd", "p_peri"]:
        fig.add_scatter(
            x=t,
            y=convert(outputs[col], to="mmHg"),
            name=col,
            legendgroup=group,
            row=4,
            col=1,
            line_color=colour,
        )

    fig.add_scatter(
        x=t,
        y=convert(outputs["v_pcd"], to="ml"),
        name="v_pcd",
        legendgroup=group,
        row=4,
        col=3,
        line_color=colour,
    )

    fig.add_scatter(
        x=t,
        y=outputs["e_t"],
        name="e_t",
        legendgroup=group,
        row=5,
        col=1,
        line_color=colour,
    )

    fig.add_scatter(
        x=t,
        y=convert(outputs["v_spt"], to="ml"),
        name="v_spt",
        legendgroup=group,
        row=5,
        col=3,
        line_color=colour,
    )
    return fig


def plot_resp(t, outputs, fig=None, colour=qualitative.Plotly[0], group=None):

    channels = [
        ["x", "y"],
        ["dx_dt", "dy_dt"],
        ["v_alv", "v_th", "v_bth"],
        ["dv_alv_dt"],
        ["p_pl", "p_mus"],
    ]

    if fig is None:
        fig = make_subplots(len(channels), 1, shared_xaxes="all")
        fig.update_layout(hovermode="x")
        fig.update_yaxes(row=1, col=1, title_text="Lienard states")
        fig.update_yaxes(row=2, col=1, title_text="Lienard derivatives")
        fig.update_yaxes(row=3, col=1, title_text="Respiratory volumes")
        fig.update_yaxes(row=4, col=1, title_text="Volume derivative")
        fig.update_yaxes(row=5, col=1, title_text="Pleural pressure")
        fig.update_xaxes(row=len(channels), col=1, title_text="Time (s)")

    if group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour,
            mode="lines",
        )

    for i, chans in enumerate(channels):
        for chan in chans:
            fig.add_scatter(
                x=t,
                y=outputs[chan],
                name=chan,
                legendgroup=group,
                showlegend=group is None,
                row=i + 1,
                col=1,
                line_color=colour,
            )

    return fig


def plot_states(t, outputs, fig=None, colour=qualitative.Plotly[0], group=None):

    plot_spec = [
        [state, f"d{state}_dt"] for state in ["v_pa", "v_pu", "v_lv", "v_ao", "v_vc", "v_rv"]
    ]

    if fig is None:
        fig = make_subplots(len(plot_spec), len(plot_spec[0]), shared_xaxes="all")
        fig.update_layout(hovermode="x")
        for i_row, row in enumerate(plot_spec):
            for i_col, channel in enumerate(row):
                fig.update_yaxes(row=i_row + 1, col=i_col + 1, title_text=channel)

    if group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour,
            mode="lines",
        )

    for i_row, row in enumerate(plot_spec):
        for i_col, channel in enumerate(row):
            fig.add_scatter(
                x=t,
                y=outputs[channel],
                name=channel,
                legendgroup=group,
                showlegend=group is None,
                row=i_row + 1,
                col=i_col + 1,
                line_color=colour,
            )

    return fig
