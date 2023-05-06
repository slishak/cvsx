from itertools import cycle

import numpy as np
from plotly.subplots import make_subplots
from plotly.colors import qualitative

from cvsx.unit_conversions import convert


def latex(s: str):
    if s.startswith("p_") or s.startswith("q_") or s.startswith("v_"):
        s = s[0].upper() + s[1:]
    parts = []
    for substr in s.split(" "):
        if "_" in substr:
            substr = substr.replace("_", "_{\\rm ") + "}"
        if substr.startswith("("):
            substr = rf"\text{{ {substr} }}"
        parts.append(substr)
    s = " ".join(parts)
    return rf"$ \small{{ {s} }}$"


def _plot_vertical_grid(
    y_labels,
    channels,
    units,
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    accumulate_max=False,
    **kwargs,
):
    if fig is None:
        fig = make_subplots(len(y_labels), 1, shared_xaxes="all")
        fig.update_layout(hovermode="x")
        for i, (label, unit) in enumerate(zip(y_labels, units)):
            if unit is None:
                title = label
            else:
                title = f"{label} ({unit})"
            fig.update_yaxes(row=i + 1, col=1, title_text=title, minor_showgrid=True)
        fig.update_xaxes(row=len(y_labels), col=1, title_text="Time (s)")

    if showlegend and group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour or "black",
            line_dash=dash or "solid",
            mode="lines",
            opacity=1.0 if dash is None else 0.0,
            **kwargs,
        )

    colours = cycle(qualitative.Plotly)

    for i, chans in enumerate(channels):
        dashes = cycle(["solid", "dash", "dot", "dashdot"])
        for chan in chans:
            try:
                y = outputs[chan]
            except KeyError:
                print(f"Missing channel: {chan}")
                continue

            if units[i] is not None:
                y = convert(y, to=units[i])

            fig.add_scatter(
                x=t,
                y=y,
                name=latex(chan),
                legendgroup=group,
                showlegend=(
                    (group is None and dash is None) or len(chans) > 1 or showlegend == "all"
                ),
                row=i + 1,
                col=1,
                line_color=colour or next(colours),
                line_dash=dash or next(dashes),
                mode=mode,
                **kwargs,
            )

            if accumulate_max:
                fig.add_scatter(
                    x=t,
                    y=np.maximum.accumulate(y),
                    name=chan,
                    legendgroup=group,
                    showlegend=(
                        (group is None and dash is None) or len(chans) > 1 or showlegend == "all"
                    ),
                    row=i + 1,
                    col=1,
                    line_color=colour or next(colours),
                    line_dash=dash or next(dashes),
                    mode=mode,
                    **kwargs,
                )

    return fig


def plot_lv_pressures(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    y_labels = ["Pressure", "Flow rates"]

    channels = [
        ["p_lv", "p_ao", "p_pu"],
        ["q_mt", "q_av", "q_sys"],
    ]

    units = ["mmHg", "ml/s"]

    return _plot_vertical_grid(
        y_labels,
        channels,
        units,
        t,
        outputs,
        fig,
        colour,
        dash,
        group,
        mode,
        showlegend,
        **kwargs,
    )


def plot_rv_pressures(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    y_labels = ["Pressure", "Flow rates"]

    channels = [
        ["p_rv", "p_pa", "p_vc"],
        ["q_tc", "q_pv", "q_pul"],
    ]

    units = [
        "mmHg",
        "ml/s",
    ]

    return _plot_vertical_grid(
        y_labels, channels, units, t, outputs, fig, colour, dash, group, mode, showlegend, **kwargs
    )


def plot_vent_interaction(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    specs = [
        [{}, {"rowspan": 3}],
        [{"secondary_y": True}, None],
        [{}, None],
    ]

    if fig is None:
        fig = make_subplots(3, 2, shared_xaxes="columns", specs=specs)
        fig.update_xaxes(row=3, col=1, title_text="Time (s)")
        fig.update_yaxes(row=1, col=1, title_text="Ventricle volume (ml)", minor_showgrid=True)
        fig.update_yaxes(row=2, col=1, title_text=latex("e(t)"), minor_showgrid=True)
        fig.update_yaxes(row=3, col=1, title_text=latex("v_spt (ml)"), minor_showgrid=True)
        fig.update_yaxes(col=2, title_text="Ventricle pressure (mmHg)", minor_showgrid=True)
        fig.update_xaxes(col=2, title_text="Ventricle volume (ml)", minor_showgrid=True)

    if showlegend and group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour or "black",
            line_dash=dash or "solid",
            mode="lines",
            **kwargs,
        )

    fig.add_scatter(
        x=t,
        y=convert(outputs["v_lv"], to="ml"),
        name="Left ventricle",
        legendgroup=group,
        showlegend=True,
        row=1,
        col=1,
        line_color=colour or qualitative.Plotly[0],
        line_dash=dash or "solid",
        mode=mode,
        **kwargs,
    )
    fig.add_scatter(
        x=t,
        y=convert(outputs["v_rv"], to="ml"),
        name="Right ventricle",
        legendgroup=group,
        showlegend=True,
        row=1,
        col=1,
        line_color=colour or qualitative.Plotly[1],
        line_dash=dash or "dash",
        mode=mode,
        **kwargs,
    )
    fig.add_scatter(
        x=t,
        y=outputs["e_t"],
        name=latex("e(t)"),
        legendgroup=group,
        showlegend=False,
        row=2,
        col=1,
        line_color=colour or "black",
        line_dash=dash or "solid",
        mode=mode,
        **kwargs,
    )
    p_pl_plotted = False
    try:
        fig.add_scatter(
            x=t,
            y=outputs["p_pl"],
            name=latex("p_pl"),
            legendgroup=group,
            showlegend=False,
            row=2,
            col=1,
            line_color=colour or "red",
            line_dash=dash or "solid",
            mode=mode,
            secondary_y=True,
            **kwargs,
        )
    except KeyError:
        pass
    else:
        p_pl_plotted = True
        fig.update_yaxes(
            row=2,
            col=1,
            title_text=latex("p_pl (mmHg)"),
            title_font_color="red",
            secondary_y=True,
        )

    if not p_pl_plotted:
        try:
            fig.add_scatter(
                x=t,
                y=outputs["ds_dt"] * 60,
                name=latex("HR(t) (bpm)"),
                legendgroup=group,
                showlegend=False,
                row=2,
                col=1,
                line_color=colour or "red",
                line_dash=dash or "solid",
                mode=mode,
                secondary_y=True,
                **kwargs,
            )
        except KeyError:
            pass
        else:
            fig.update_yaxes(
                row=2,
                col=1,
                title_text=latex("HR(t) (bpm)"),
                title_font_color="red",
                secondary_y=True,
            )

    fig.add_scatter(
        x=t,
        y=convert(outputs["v_spt"], to="ml"),
        name=latex("v_spt"),
        legendgroup=group,
        showlegend=False,
        row=3,
        col=1,
        line_color=colour or "black",
        line_dash=dash or "solid",
        mode=mode,
        **kwargs,
    )
    fig.add_scatter(
        x=convert(outputs["v_lv"], to="ml"),
        y=convert(outputs["p_lv"], to="mmHg"),
        name="Left",
        legendgroup=group,
        showlegend=False,
        row=1,
        col=2,
        line_color=colour or qualitative.Plotly[0],
        line_dash=dash or "solid",
        mode=mode,
        **kwargs,
    )
    fig.add_scatter(
        x=convert(outputs["v_rv"], to="ml"),
        y=convert(outputs["p_rv"], to="mmHg"),
        name="Right",
        legendgroup=group,
        showlegend=False,
        row=1,
        col=2,
        line_color=colour or qualitative.Plotly[1],
        line_dash=dash or "dash",
        mode=mode,
        **kwargs,
    )

    return fig


def plot_outputs(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):
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

    if showlegend and group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour or "black",
            line_dash=dash or "solid",
            mode="lines",
            **kwargs,
        )

    for col in ["p_lvf", "p_lv", "p_ao", "p_pu", "p_aom", "p_aos", "p_aod"]:
        try:
            fig.add_scatter(
                x=t,
                y=convert(outputs[col], to="mmHg"),
                name=col,
                legendgroup=group,
                showlegend=group is None,
                row=1,
                col=1,
                line_color=colour,
                mode=mode,
                **kwargs,
            )
        except KeyError:
            pass

    for col in ["v_lvf", "v_lv", "v_ao", "v_pu"]:
        fig.add_scatter(
            x=t,
            y=convert(outputs[col], "l", "ml"),
            name=col,
            legendgroup=group,
            showlegend=group is None,
            row=1,
            col=3,
            line_color=colour,
            mode=mode,
            **kwargs,
        )

    for col in ["p_rvf", "p_rv", "p_pa", "p_vc", "p_vcm"]:
        try:
            fig.add_scatter(
                x=t,
                y=convert(outputs[col], to="mmHg"),
                name=col,
                legendgroup=group,
                showlegend=group is None,
                row=2,
                col=1,
                line_color=colour,
                mode=mode,
                **kwargs,
            )
        except KeyError:
            pass

    for col in ["v_rvf", "v_rv", "v_pa", "v_vc"]:
        fig.add_scatter(
            x=t,
            y=convert(outputs[col], to="ml"),
            name=col,
            legendgroup=group,
            showlegend=group is None,
            row=2,
            col=3,
            line_color=colour,
            mode=mode,
            **kwargs,
        )

    for col in ["q_mt", "q_av", "q_tc", "q_pv", "q_pul", "q_sys"]:
        fig.add_scatter(
            x=t,
            y=outputs[col],
            name=col,
            legendgroup=group,
            showlegend=group is None,
            row=3,
            col=1,
            line_color=colour,
            mode=mode,
            **kwargs,
        )

    fig.add_scatter(
        x=convert(outputs["v_lv"], to="ml"),
        y=convert(outputs["p_lv"], to="mmHg"),
        name="lv",
        legendgroup=group,
        showlegend=group is None,
        row=3,
        col=3,
        line_color=colour,
        mode=mode,
        **kwargs,
    )
    fig.add_scatter(
        x=convert(outputs["v_rv"], to="ml"),
        y=convert(outputs["p_rv"], to="mmHg"),
        name="rv",
        legendgroup=group,
        showlegend=group is None,
        row=3,
        col=4,
        line_color=colour,
        mode=mode,
        **kwargs,
    )
    for col in ["p_pcd", "p_peri"]:
        fig.add_scatter(
            x=t,
            y=convert(outputs[col], to="mmHg"),
            name=col,
            legendgroup=group,
            showlegend=group is None,
            row=4,
            col=1,
            line_color=colour,
            mode=mode,
            **kwargs,
        )

    fig.add_scatter(
        x=t,
        y=convert(outputs["v_pcd"], to="ml"),
        name="v_pcd",
        legendgroup=group,
        showlegend=group is None,
        row=4,
        col=3,
        line_color=colour,
        mode=mode,
        **kwargs,
    )

    fig.add_scatter(
        x=t,
        y=outputs["e_t"],
        name="e_t",
        legendgroup=group,
        showlegend=group is None,
        row=5,
        col=1,
        line_color=colour,
        mode=mode,
        **kwargs,
    )

    fig.add_scatter(
        x=t,
        y=convert(outputs["v_spt"], to="ml"),
        name="v_spt",
        legendgroup=group,
        showlegend=group is None,
        row=5,
        col=3,
        line_color=colour,
        mode=mode,
        **kwargs,
    )
    return fig


def plot_resp(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    y_labels = [
        "Lienard states",
        "Lienard derivatives",
        "Respiratory volumes",
        "Volume derivative",
        "Pleural pressure",
    ]

    channels = [
        ["x", "y"],
        ["dx_dt", "dy_dt"],
        ["v_alv", "v_th", "v_bth"],
        ["dv_alv_dt"],
        ["p_pl", "p_mus"],
    ]

    units = [
        None,
        None,
        "ml",
        "ml/s",
        "mmHg",
    ]

    return _plot_vertical_grid(
        y_labels, channels, units, t, outputs, fig, colour, dash, group, mode, showlegend, **kwargs
    )


def plot_spt_resp(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    y_labels = ["Ventricle volume", latex("p_pl"), latex("v_spt")]

    channels = [
        ["v_lv", "v_rv"],
        ["p_pl"],
        ["v_spt"],
    ]

    units = [
        "ml",
        "mmHg",
        "ml",
    ]

    return _plot_vertical_grid(
        y_labels, channels, units, t, outputs, fig, colour, dash, group, mode, showlegend, **kwargs
    )


def plot_states(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    plot_spec = [
        [state, f"d{state}_dt"]
        for state in [
            "v_pa",
            "v_pu",
            "v_lv",
            "v_ao",
            "v_vc",
            "v_rv",
            "q_mt",
            "q_av",
            "q_tc",
            "q_pv",
        ]
    ]

    if fig is None:
        fig = make_subplots(len(plot_spec), len(plot_spec[0]), shared_xaxes="all")
        fig.update_layout(hovermode="x")
        for i_row, row in enumerate(plot_spec):
            for i_col, channel in enumerate(row):
                fig.update_yaxes(row=i_row + 1, col=i_col + 1, title_text=channel)

    if showlegend and group is not None:
        fig.add_scatter(
            x=[None],
            y=[None],
            name=group,
            legendgroup=group,
            showlegend=True,
            row=1,
            col=1,
            line_color=colour,
            mode="lines+markers",
            **kwargs,
        )

    for i_row, row in enumerate(plot_spec):
        for i_col, channel in enumerate(row):
            try:
                fig.add_scatter(
                    x=t,
                    y=outputs[channel],
                    name=channel,
                    legendgroup=group,
                    showlegend=group is None,
                    row=i_row + 1,
                    col=i_col + 1,
                    line_color=colour,
                    mode=mode,
                    **kwargs,
                )
            except KeyError:
                pass

    return fig


def plot_drift(
    t,
    outputs,
    fig=None,
    colour=qualitative.Plotly[0],
    dash=None,
    group=None,
    mode="lines",
    showlegend=True,
    **kwargs,
):

    y_labels = [
        latex("p_pa"),
        latex("p_pl"),
        latex("v_alv"),
    ]

    channels = [
        ["p_pa"],
        ["p_pl"],
        ["v_alv"],
    ]

    units = [
        "mmHg",
        "mmHg",
        "ml",
    ]

    return _plot_vertical_grid(
        y_labels,
        channels,
        units,
        t,
        outputs,
        fig,
        colour,
        dash,
        group,
        mode,
        showlegend,
        # accumulate_max=True,
        **kwargs,
    )
