
from config.config_managers import (BarConfig, ColorMap, MatrixConfig,
                                    ScatterConfig, ScatterWidthConfig,
                                    ViolinConfig)
from config.enums import (CorrelationColumns, CustomAnalysisConfig,
                          ErrorRateColumns, TagAmbiguityColumns,
                          TokenVariabilityColumns)
from managers.tabs.tab_managers import (BasePlotting,
                                        create_correlation_matrix_plot,
                                        create_histogram_plot,
                                        create_scatter_plot,
                                        create_scatter_width_plot,
                                        create_violin_plot)


import logging
import numpy as np
from dash import html
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import html, dcc

from config.config_managers import (
    ColorMap
)


# TODO Crete an abstract class that formats the title a nd cusomtised it based on the variant 
HEADER_BG = "#3DAFA8"


def render_dataset_stats_table(df: pd.DataFrame) -> html.Div:
    disp = df.copy()

    INT_METRICS   = ["Total Words", "Unique Words", "NE Words", "Unique NE Words"]
    RATIO_METRICS = ["NE Proportion", "TWR", "TEWR", "TTR", "TETR", "NE Type Proportion", "Tokens NE Proportion"]  # include others if present

    # --- helpers ---
    def _fmt_int_series(s: pd.Series) -> pd.Series:
        # if already strings w/ commas, parse back to numbers safely
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
        return s.map(lambda x: f"{int(x):,}" if pd.notna(x) else "")

    def _fmt_ratio_series(s: pd.Series, as_percent: bool = True, digits: int = 4) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        if as_percent:
            return s.map(lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "")
        return s.map(lambda x: f"{float(x):.{digits}f}" if pd.notna(x) else "")

    # --- format integers with commas ---
    for m in INT_METRICS:
            if m in disp.index:
                disp.loc[m] = _fmt_int_series(disp.loc[m])


    # # --- format ratios ---
    # # If you prefer decimals (e.g., 0.1054), set as_percent=False
    # if "NE Proportion" in disp.index:
    #     disp.loc["NE Proportion"] = _fmt_ratio_series(disp.loc["NE Proportion"], digits=4)
    # if "NE Type Proportion" in disp.index:
    #     disp.loc["NE Type Proportion"] = _fmt_ratio_series(disp.loc["NE Type Proportion"], digits=4)
    # if "TWR" in disp.index:
    #     disp.loc["TWR"] = _fmt_ratio_series(disp.loc["TWR"], digits=4)
    # if "TEWR" in disp.index:
    #     disp.loc["TEWR"] = _fmt_ratio_series(disp.loc["TEWR"], digits=4)

    # --- format ratios as decimals (not percentages) ---
    for m in RATIO_METRICS:
        if m in disp.index:
            disp.loc[m] = _fmt_ratio_series(disp.loc[m], digits=4)
    

    def _cid(c0, c1): return f"{c0}|{c1}"

    columns = [{"name": ["", "Metric"], "id": "Metric"}]
    for c0, c1 in disp.columns:
        columns.append({"name": [str(c0), str(c1)], "id": _cid(c0, c1)})

    rows = []
    for metric in disp.index:
        row = {"Metric": metric}
        for c0, c1 in disp.columns:
            row[_cid(c0, c1)] = disp.loc[metric, (c0, c1)]
        rows.append(row)

    table = DataTable(
        id="dataset-stats-table",
        columns=columns,
        data=rows,
        merge_duplicate_headers=True,
        style_as_list_view=True,
        style_table={"width": "100%"},
        style_cell={"textAlign": "center", "padding": "10px"},
        style_header={
            "text-align": "center",
            "background-color": HEADER_BG,
            "color": "white",
            "fontWeight": "700",
        },
        style_data_conditional=[
            {"if": {"column_id": "Metric"}, "textAlign": "left", "fontWeight": "600"},
        ],
    )
    return html.Div([table])



def plot_faceted_bar_chart(
    df: pd.DataFrame,
    metric: str,
    text: str | None,
    title: str,
    *,
    tag_order: list[str] | None = None,
    sort_bars: bool = True
) -> html.Div:
    # ---- guards & shallow copy ----
    if metric not in df.columns:
        raise KeyError(f"`metric` '{metric}' not found in DataFrame.")

    # ---- choose encodings ----
    x_col = "Tag" if "Tag" in df.columns else None
    if x_col is None:
        exclude = {metric}
        if text:
            exclude.add(text)
        cats = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]
        x_col = cats[0] if cats else df.columns[0]

    color_by  = "Tag" if "Tag" in df.columns else None
    facet_col = "Language" if "Language" in df.columns else None
    facet_row = "Split"   if "Split"   in df.columns else None

    # ---- normalise types / handle NaNs ----
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    # ---- looks like proportion? ----
    is_num = pd.api.types.is_numeric_dtype(df[metric])
    PERCENT_METRICS = {"Tokens Proportion", "Type Proportion"}
    looks_like_prop = metric in PERCENT_METRICS
    looks_like_prop = bool(is_num and df[text].ge(0).all() and df[text].le(1).all())
    y_label = "Proportion" if looks_like_prop else metric
    x_label = "Entity Tag" if x_col == "Tag" else x_col
    labels = {metric: y_label, x_col: x_label}

    # ---- category order (optional, keeps legends stable) ----
    category_orders = {}
    if x_col == "Tag" and tag_order:
        seen = list(pd.Index(tag_order).intersection(df["Tag"].unique()))
        missing = [t for t in tag_order if t not in seen]
        category_orders["Tag"] = seen + missing
        df["Tag"] = pd.Categorical(df["Tag"], categories=category_orders["Tag"], ordered=True)

    # ---- sort bars ----
    if sort_bars and not category_orders.get("Tag"):
        group_keys = [k for k in [facet_row, facet_col] if k]
        if group_keys:
            orders = []
            for _, g in df.groupby(group_keys, dropna=False):
                order = g.groupby(x_col, as_index=False)[metric].mean().sort_values(metric, ascending=False)[x_col].tolist()
                orders.append(order)
            order = max(orders, key=len) if orders else None
        else:
            order = df.groupby(x_col, as_index=False)[metric].mean().sort_values(metric, ascending=False)[x_col].tolist()
        if order:
            category_orders[x_col] = order
            df[x_col] = pd.Categorical(df[x_col], categories=order, ordered=True)

    # ---- colors (fallback if ColorMap() isn’t in scope) ----
    try:
        cmap = ColorMap().color_map  # your existing helper, if available
    except NameError:
        cmap = {}  # let Plotly assign defaults


    # ---- build figure ----
    fig = px.bar(
        df,
        x=x_col,
        y=metric,
        color=color_by if color_by in df.columns else None,
        facet_col=facet_col,
        facet_row=facet_row,
        facet_row_spacing=0.01,                 # add breathing room
        facet_col_spacing=0.06,
        category_orders=category_orders or None,
        color_discrete_map=cmap if cmap else None,
        text=text if (text and text in df.columns) else None,
        title=title,
        labels=labels,
    )

    # ---- text labels formatting ----
    if text and text in df.columns:
        if pd.api.types.is_numeric_dtype(df[text]):
            if looks_like_prop:
                # proportions → always 2 decimals
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                    cliponaxis=False
                )
            else:
                # other numbers → raw values
                fig.update_traces(
                    texttemplate="%{text}",
                    textposition="outside",
                    cliponaxis=False
                )
        else:
            fig.update_traces(textposition="outside", cliponaxis=False)
    elif looks_like_prop:
        # show y as proportion with 2 decimals
        fig.update_traces(
            texttemplate="%{y:.2f}", 
            textposition="outside", 
            cliponaxis=False
        )

    # ---- axes & layout ----
    if looks_like_prop:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", matches=None)
    else:
        fig.update_yaxes(rangemode="tozero", matches=None)

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=20, r=20, b=20),
        legend_title=None,
        height=800,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        bargap=0.25,
    )

    # ---- adjust facet titles (move away from bars) ----
    fig.for_each_annotation(lambda a: a.update(
        # text=a.text.replace("Dataset=", "").replace("Split=", ""),  # optional cleanup
        # y=a.y + 0.1,        # push higher up
        font=dict(size=14), # slightly larger for readability
        xanchor="center",
        yanchor="bottom"
    ))

    # ---- rotate x tick labels ----
    fig.update_xaxes(tickangle=-30)

    return html.Div([dcc.Graph(figure=fig)])



def plot_entity_span_distribution(
    df: pd.DataFrame,
    metric: str,
    text: str | None,
    title: str,
    *,
    tag_order: list[str] | None = None,           # e.g. _DEFAULT_ENTITY_SPANS
) -> html.Div:
    """
    Faceted bar chart for entity *span* distributions.

    - x = 'Entity'
    - y = metric (e.g. 'Span Proportion' or 'Span Count')
    - bar labels = text (column name or None)
    - facets: 'Scheme' (cols) × 'Split' (rows)
    - color = 'Language'
    - tag_order enforces consistent ordering of entities
    """

    # Make splits look nice
    split_mapping = {"train": "Train", "test": "Test"}
    if "Split" in df.columns:
        df["Split"] = df["Split"].replace(split_mapping)

    # Detect proportion for y-axis formatting
    looks_like_prop = (
        pd.api.types.is_numeric_dtype(df[metric])
        and df[metric].dropna().between(0.0, 1.0).all()
    )
    y_label = "Percentage" if looks_like_prop else metric

    fig = px.bar(
        df,
        x="Entity",
        y=metric,
        color="Language",
        text=text if text and text in df.columns else None,
        barmode="group",
        facet_col="Scheme",
        facet_row="Split",
        category_orders={
            "Scheme": ["IOB1", "IOB2"],
            "Entity": tag_order,
        },
        labels={"Entity": "Entity Span", metric: y_label},
    )

    # Format text on bars
    if text and text in df.columns:
        if pd.api.types.is_numeric_dtype(df[text]):
            fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        else:
            fig.update_traces(textposition="outside", cliponaxis=False)

    # Y-axis formatting
    if looks_like_prop:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    else:
        fig.update_yaxes(rangemode="tozero")

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700,
        margin=dict(t=60, l=20, r=20, b=20),
        legend_title=None,
    )

    # Clean up facet labels
    # fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Scheme=", "").replace("Split=", "")))

    return html.Div([dcc.Graph(figure=fig)])




def plot_entity_span_complexity_box(
    df: pd.DataFrame,
    *,
    title: str = "Entity span length distribution",
    tag_order: list[str] | None = None,
    show_mean_labels: bool = False,
) -> html.Div:

    if "Split" in df.columns:
        df["Split"] = df["Split"].replace({"train": "Train", "test": "Test"})

    # Base: faceted box plot
    fig = px.box(
        df,
        x="Entity",
        y="Span Length",
        color="Language",
        facet_col="Scheme",
        facet_row="Split",
        points=False,
        facet_col_spacing=0.15,
        category_orders={"Scheme": ["IOB1", "IOB2"], "Entity": tag_order},
        labels={"Entity": "Entity Span", "Span Length": "Span Length Distribution"},
        title=title,
    )

    # Show mean marker on the boxes themselves
    fig.update_traces(boxmean="sd", selector=dict(type="box"))

    if show_mean_labels:
        # Compute means per facet group and overlay as a faceted scatter
        means = (
            df.groupby(["Language", "Split", "Scheme", "Entity"], as_index=False)["Span Length"]
                .mean()
                .rename(columns={"Span Length": "Mean"})
        )
        means["Mean Label"] = means["Mean"].round(1).astype(str)

        mean_fig = px.scatter(
            means,
            x="Entity",
            y="Mean",
            color="Language",
            text="Mean Label",
            facet_col="Scheme",
            facet_row="Split",
            category_orders={"Scheme": ["IOB1", "IOB2"], "Entity": tag_order},
        )
        mean_fig.update_traces(
            marker_symbol="diamond",
            marker_size=10,
            textposition="top center",
            showlegend=False,            # don’t duplicate legends
            hovertemplate="Entity=%{x}<br>Mean=%{y:.2f}<br>Language=%{marker.color}<extra></extra>",
        )

        # Merge the scatter traces (they inherit the correct facets automatically)
        fig.add_traces(mean_fig.data)

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=20, r=20, b=20),
        legend_title=None,
        height=700,
    )
    # fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Scheme=", "").replace("Split=", "")))

    return html.Div([dcc.Graph(figure=fig)]) 


def plot_colored_bar_chart(
    df: pd.DataFrame,
    *,
    metric: str = "Standard Deviation",
    title: str = "Tag type variability",
    tag_order: list[str] | None = None,
    show_values: bool = True,
) -> html.Div:
    """
    Simple grouped bar chart:
      x = Tag
      y = metric (e.g., 'Standard Deviation')
      color = Language
      pattern_shape = Split (so Train/Test are distinguishable without facets)
    """
    

    # nice labels for split if needed
    if "Split" in df.columns:
        df["Split"] = df["Split"].replace({"train":"Train", "test":"Test"})

    # enforce tag order if provided
    if "Tag" in df.columns and tag_order:
        cats = [t for t in tag_order if t in df["Tag"].unique().tolist()]
        df["Tag"] = pd.Categorical(df["Tag"], categories=cats, ordered=True)

    fig = px.bar(
        df,
        x="Tag",
        y=metric,
        color="Language",
        barmode="group",
        text=metric if show_values else None,
        facet_row='Split',
        category_orders={"Tag": list(df["Tag"].cat.categories) if "Tag" in df.columns and hasattr(df["Tag"], "cat") else None},
        title=title,
        labels={"Tag": "Entity Tag", metric: metric},
    )

    if show_values:
        # compact numeric labels
        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False)

    # enforce 2 decimals on y-axis ticks + hover
    fig.update_yaxes(tickformat=".2f")
    fig.update_traces(
        hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=20, r=20, b=40),
        legend_title=None,
        bargap=0.25,
        height=800,
    )
    fig.update_xaxes(tickangle=-30)
    return html.Div([dcc.Graph(figure=fig)])


def render_oov_row_table(
    df: pd.DataFrame,
    *,
    language_col: str | None = None,
    oov_count_col: str | None = None,
    oov_rate_col: str | None = None,
    total_test_col: str | None = None,
    digits: int = 4,
) -> html.Div:
    """
    Renders a single-row-per-language OOV table.
    Auto-detects columns if not provided. Supports both 'Words' and 'Core Tokens' naming.

    Expected-ish columns (any of these; first found wins):
      language: ["Language", "Corpus", "DS"]
      oov_count: ["OOV Words Count", "OOV Core Tokens Count", "OOV Count"]
      oov_rate:  ["OOV Rate", "OOV Proportion"]
      total:     ["Total Unique Words in Test", "Total Unique Core Tokens in Test", "Test Unique Count"]
    """

    def _pick(colnames: list[str], fallback: str | None = None) -> str | None:
        lower_map = {c.lower(): c for c in df.columns}
        for name in colnames:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return fallback

    language_col = language_col or _pick(["Language", "Corpus", "DS"])
    oov_count_col = oov_count_col or _pick(["OOV Words Count", "OOV Core Tokens Count", "OOV Count"])
    oov_rate_col  = oov_rate_col  or _pick(["OOV Rate", "OOV Proportion"])
    total_test_col = total_test_col or _pick(
        ["Total Unique Words in Test", "Total Unique Core Tokens in Test", "Test Unique Count"]
    )

    # Build a view with whatever we have
    wanted = [c for c in [language_col, oov_count_col, oov_rate_col, total_test_col] if c]
    disp = df[wanted].copy()
    # Formatters
    def _fmt_int_like(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s): s = pd.to_numeric(s, errors="coerce")
        return s.map(lambda x: f"{int(round(float(x))):,}" if pd.notna(x) else "")

    def _fmt_ratio(s: pd.Series, nd: int) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s): s = pd.to_numeric(s, errors="coerce")
        return s.map(lambda v: f"{float(v):.{nd}f}" if pd.notna(v) else "")

    if oov_count_col in disp.columns:
        disp[oov_count_col] = _fmt_int_like(disp[oov_count_col])
    if total_test_col in disp.columns:
        disp[total_test_col] = _fmt_int_like(disp[total_test_col])
    if oov_rate_col in disp.columns:
        disp[oov_rate_col] = _fmt_ratio(disp[oov_rate_col], digits)

    # Columns (preserve a friendly order)
    columns = []
    if language_col:      columns.append({"name": "Language", "id": language_col})
    if oov_count_col:    columns.append({"name": oov_count_col, "id": oov_count_col})
    if oov_rate_col:     columns.append({"name": oov_rate_col, "id": oov_rate_col})
    if total_test_col:   columns.append({"name": total_test_col, "id": total_test_col})

    table = DataTable(
        id="language-oov-row-table",
        columns=columns,
        data=disp.to_dict("records"),
        style_as_list_view=True,
        style_table={"width": "100%"},
        style_cell={"textAlign": "center", "padding": "10px"},
        style_header={
            "text-align": "center",
            "background-color": HEADER_BG,
            "color": "white",
            "fontWeight": "700",
        },
        style_data_conditional=[
            {"if": {"column_id": language_col or "Language"}, "textAlign": "left", "fontWeight": "600"},
        ],
    )
    return html.Div([table])


def plot_entity_tag_oov_bar(
    df: pd.DataFrame,
    *,
    title: str = "Per-Tag OOV Rate (Train→Test)",
    tag_order: list[str] | None = None,
    show_values: bool = True,
    language_col: str | None = None,
    tag_col: str | None = None,
    oov_rate_col: str | None = None,
    percent_axis: bool = False,     # set True if you want y-axis as %
    text_digits: int = 3,
) -> html.Div:
    """
    Grouped bar of OOV rate by tag + language.
    Column auto-detects by default; can be overridden by args.
    """

    # ---- pick columns (case-insensitive) ----
    def _pick(colnames: list[str], fallback: str | None = None) -> str | None:
        lower_map = {c.lower(): c for c in df.columns}
        for name in colnames:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return fallback

    language_col = language_col or _pick(["Language", "Corpus", "DS"])
    tag_col     = tag_col     or _pick(["Tag", "Entity", "Label"])
    oov_rate_col= oov_rate_col or _pick(["OOV Rate", "OOV Proportion"])

    if not tag_col or not oov_rate_col:
        # Nothing meaningful to plot
        return html.Div([html.Div("OOV columns not found.", className="text-muted")])

    data = df.copy()

    # stable tag order if provided
    if tag_order and tag_col in data.columns:
        cats = [t for t in tag_order if t in data[tag_col].unique().tolist()]
        if cats:
            data[tag_col] = pd.Categorical(data[tag_col], categories=cats, ordered=True)

    # detect proportion 0..1
    looks_like_prop = (
        pd.api.types.is_numeric_dtype(data[oov_rate_col])
        and data[oov_rate_col].dropna().between(0.0, 1.0).all()
    )

    fig = px.bar(
        data,
        x=tag_col,
        y=oov_rate_col,
        color=language_col if language_col in data.columns else None,
        barmode="group",
        text=oov_rate_col if show_values else None,
        category_orders={tag_col: list(data[tag_col].cat.categories) if hasattr(data[tag_col], "cat") else None},
        labels={tag_col: "Entity Tag", oov_rate_col: "OOV Rate"},
        title=title,
    )

    # Text labels: show decimals only when it's a proportion
    if show_values:
        if looks_like_prop:
            # decimals (e.g., 0.382)
            fig.update_traces(texttemplate=f"%{{y:.{text_digits}f}}", textposition="outside", cliponaxis=False)
        else:
            # raw numbers fallback
            fig.update_traces(texttemplate="%{y}", textposition="outside", cliponaxis=False)

    # Y axis format
    if percent_axis and looks_like_prop:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title="OOV Rate (%)")
    else:
        fig.update_yaxes(rangemode="tozero", title="OOV Rate (Rate)" if looks_like_prop else oov_rate_col)

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=20, r=20, b=40),
        legend_title=None,
        bargap=0.25,
        height=600,
    )
    fig.update_xaxes(tickangle=-30)

    return html.Div([dcc.Graph(figure=fig)])





# def plot_overlap_heatmaps(
#     df: pd.DataFrame,
#     *,
#     title: str = "Type Overlap Across Entity Tags",
#     panel_by: str = "Language",        # or "Split" — which column to split into panels
#     filter_equals: dict | None = None, # e.g. {"Split": "Test"} to show only Test across datasets
#     tag_order: list[str] | None = None,
#     lower_triangle_only: bool = True,
#     colorscale: str = "RdBu_r",
# ) -> html.Div:
#     """
#     Render a row of heatmaps from a tidy long DataFrame with columns:
#       ['Language','Split','Tag1','Tag2','Overlap Count'].

#     Examples:
#       plot_overlap_heatmaps(df, panel_by="Language", filter_equals={"Split":"Test"})
#       plot_overlap_heatmaps(df, panel_by="Split")  # one panel per split
#     """
#     needed = {"Tag1", "Tag2", "Overlap Count"}
#     if not needed.issubset(df.columns):
#         missing = ", ".join(sorted(needed - set(df.columns)))
#         raise KeyError(f"Missing required columns: {missing}")

    

#     # Optional filtering (e.g., keep only Split=='Test')
#     if filter_equals:
#         for k, v in filter_equals.items():
#             if k in df.columns:
#                 df = df[df[k] == v]

#     # Choose panels
#     if panel_by not in df.columns:
#         print(panel_by)
#         print(df.columns)
#         # fall back to single panel
#         df["_panel"] = "All"
#         panel_col = "_panel"
#     else:
#         panel_col = panel_by

#     # Order of tags (matrix axes)
#     if tag_order is not None:
#         tags = [t for t in tag_order if t in set(df["Tag1"]).union(set(df["Tag2"]))]
#     else:
#         tags = sorted(set(df["Tag1"]).union(set(df["Tag2"])))

#     # Build a matrix per panel
#     panels = {}
#     for panel_value, g in df.groupby(panel_col, dropna=False):
#         # pivot to square matrix (fill missing with 0)
#         mat = (
#             g.pivot_table(index="Tag1", columns="Tag2", values="Overlap Count", aggfunc="sum", fill_value=0)
#             .reindex(index=tags, columns=tags, fill_value=0)
#         )
#         if lower_triangle_only:
#             mask = np.tril(np.ones_like(mat, dtype=bool))
#             mat = mat.where(mask)  # keep lower triangle (diagonal kept as-is)
#         panels[str(panel_value)] = mat

#     # Shared max for consistent color scale
#     max_value = max(int(m.to_numpy()[~np.isnan(m.to_numpy())].max()) if m.size else 0 for m in panels.values())

#     # Compose subplots
#     fig = make_subplots(
#         rows=1,
#         cols=len(panels),
#         subplot_titles=list(panels.keys()),
#         horizontal_spacing=0.1,
#     )

#     # Add each panel’s heatmap
#     for col_idx, (name, mat) in enumerate(panels.items(), start=1):
#         # text labels (blank for NaNs from masking)
#         text_data = np.where(mat.isnull(), "", mat.fillna(0).astype(int).astype(str))
#         fig.add_trace(
#             go.Heatmap(
#                 z=mat,
#                 x=mat.columns,
#                 y=mat.index,
#                 coloraxis="coloraxis",
#                 text=text_data,
#                 texttemplate="%{text}",
#                 hoverinfo="text+z",
#                 showscale=False,  # we’ll show a single shared colorbar
#             ),
#             row=1, col=col_idx
#         )

#     # Layout & shared color scale
#     fig.update_layout(
#         title=title,
#         template="plotly_white",
#         height=600,
#         width=1200,
#         coloraxis=dict(
#             colorscale=colorscale,
#             cmin=0,
#             cmax=max_value,
#             colorbar=dict(title="Counts", len=0.9),
#         ),
#         margin=dict(t=60, l=20, r=20, b=20),
#     )
#     # show colorbar on the last trace
#     if fig.data:
#         fig.data[-1].update(showscale=True)

#     # Clean grids
#     for ax in fig.layout:
#         if ax.startswith("xaxis") or ax.startswith("yaxis"):
#             fig.layout[ax].update(showgrid=False)

#     return html.Div([dcc.Graph(figure=fig)])


def plot_overlap_heatmaps(
    df: pd.DataFrame,
    *,
    title: str = "Type Overlap Across Entity Tags",
    panel_by: str = "Language",        # "Language" | "Split"
    filter_equals: dict | None = None, # e.g. {"Split": "Test"}
    tag_order: list[str] | None = None,
    lower_triangle_only: bool = True,
    colorscale: str = "RdBu_r",
) -> html.Div:
    # required core columns
    needed = {"Tag1", "Tag2", "Overlap Count"}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        raise KeyError(f"Missing required columns: {missing}. Seen: {list(df.columns)}")

    # optional filter(s)
    if filter_equals:
        for k, v in filter_equals.items():
            if k in df.columns:
                df = df[df[k] == v]

    if df.empty:
        return html.Div(dcc.Markdown("**No data to plot** after filters."), className="text-muted")

    # resolve panel column robustly
    if panel_by in df.columns:
        panel_col = panel_by
    else:
        # try common aliases to avoid “All” collapse
        for alias in ("Language", "Dataset", "Split"):
            if alias in df.columns:
                panel_col = alias
                break
        else:
            # no usable facet column — tell the user what we saw
            return html.Div(
                dcc.Markdown(
                    f"**Can't facet**: requested `panel_by='{panel_by}'`, but none of "
                    "`Language`, `Dataset`, or `Split` exist.\n\n"
                    f"Available columns: `{list(df.columns)}`"
                ),
                className="text-danger"
            )

    # axis tag order
    if tag_order is not None:
        all_tags = set(df["Tag1"]).union(df["Tag2"])
        tags = [t for t in tag_order if t in all_tags]
    else:
        tags = sorted(set(df["Tag1"]).union(df["Tag2"]))

    # build matrices
    panels = {}
    for val, g in df.groupby(panel_col, dropna=False):
        mat = (g.pivot_table(index="Tag1", columns="Tag2",
                             values="Overlap Count", aggfunc="sum", fill_value=0)
                 .reindex(index=tags, columns=tags, fill_value=0))
        if lower_triangle_only:
            mask = np.tril(np.ones_like(mat, dtype=bool))
            mat = mat.where(mask)
        panels[str(val)] = mat

    if not panels:
        return html.Div(dcc.Markdown("**No panels produced**."), className="text-muted")

    # shared color scale max (avoid cmax=0)
    def _max(m):
        arr = m.to_numpy()
        arr = arr[~np.isnan(arr)]
        return int(arr.max()) if arr.size else 0
    cmax = max((_max(m) for m in panels.values()), default=0) or 1

    fig = make_subplots(
        rows=1, cols=len(panels),
        subplot_titles=list(panels.keys()),
        horizontal_spacing=0.1,
    )

    for i, (name, mat) in enumerate(panels.items(), start=1):
        text = np.where(mat.isnull(), "", mat.fillna(0).astype(int).astype(str))
        fig.add_trace(
            go.Heatmap(
                z=mat.values,
                x=mat.columns.tolist(),
                y=mat.index.tolist(),
                coloraxis="coloraxis",
                text=text,
                texttemplate="%{text}",
                hoverinfo="text+z",
                showscale=False,
            ),
            row=1, col=i
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        width=max(650, 800 * len(panels)),  # scale with #panels
        coloraxis=dict(
            colorscale=colorscale,
            cmin=0, cmax=cmax,
            colorbar=dict(title="Counts", len=0.9),
        ),
        margin=dict(t=60, l=20, r=20, b=20),
    )
    if fig.data:
        fig.data[-1].update(showscale=True)

    # remove gridlines
    for ax in fig.layout:
        if isinstance(ax, str) and (ax.startswith("xaxis") or ax.startswith("yaxis")):
            fig.layout[ax].update(showgrid=False)

    return html.Div([dcc.Graph(figure=fig)])


def plot_token_behaviour_bar(
    df: pd.DataFrame,
    *,
    title: str,
    # Column names (auto-detected if None)
    tag_col: str | None = None,         # e.g. "Tag" / "True Labels" / "Entity"
    metric_col: str | None = None,      # e.g. "Mean Value" / "Tokenization Rate"
    level_col: str | None = None,       # e.g. "Level" (Token/Word OR Consistency/Inconsistency)
    text_col: str | None = None,        # e.g. "Text Label"
    color_col: str | None = None,       # e.g. "Language" or "Language"
    split_col: str | None = None,       # e.g. "Split"
    # Facet flags
    facet_by_level: bool = False,       # facet rows by Level
    facet_by_split: bool = False,       # facet cols by Split
    # NEW: explicit facet columns (row/col). If set, these override the flags above.
    facet_row_col: str | None = None,   # ← NEW
    facet_col_col: str | None = None,   # ← NEW
    # Visual tuning
    tag_order: list[str] | None = None,
    level_order: list[str] | None = None,
    decimals: int = 3,
    y_axis_label: str = "Average Score",
    height: int | None = None,
) -> html.Div:
    """
    Unified bar plotter for token-level behavioural metrics:
      - Tokenisation Rate → facet_by_level=False, facet_by_split=False
      - Ambiguity → facet_by_level=True (Token vs Word levels)
      - Consistency → facet_by_level=True (Consistency vs Inconsistency levels)

    Always shows Y as decimal values (no %).
    """

    if df is None or df.empty:
        return html.Div([html.Div("No data to plot.", className="text-muted")])

    data = df.copy()
    cols = set(data.columns)

    def _pick(candidates, fb=None):
        for c in candidates:
            if c in cols:
                return c
        return fb

    # ---- Auto-detect columns ----
    tag_col    = tag_col    or _pick(["Tag", "True Labels", "Entity"])
    metric_col = metric_col or _pick(["Mean Value", "Tokenization Rate", "Tokenisation Rate", "value", "score"])
    level_col  = level_col  or _pick(["Level"])
    text_col   = text_col   or _pick(["Text Label"])
    color_col  = color_col  or _pick(["Dataset", "Language"])
    split_col  = split_col  or _pick(["Split"])

    if not tag_col or not metric_col:
        return html.Div([html.Div("Required columns not found (Tag/Metric).", className="text-muted")])

    # ---- Ordering ----
    if tag_order and tag_col in data.columns:
        cats = [t for t in tag_order if t in data[tag_col].astype(str).unique().tolist()]
        if cats:
            data[tag_col] = pd.Categorical(data[tag_col].astype(str), categories=cats, ordered=True)

    if level_order and level_col and level_col in data.columns:
        levs = [l for l in level_order if l in data[level_col].astype(str).unique().tolist()]
        if levs:
            data[level_col] = pd.Categorical(data[level_col].astype(str), categories=levs, ordered=True)

    # ---- Faceting logic ----
    # default behaviour (old flags)
    facet_row = level_col if (facet_by_level and level_col in data.columns) else None
    facet_col = split_col if (facet_by_split and split_col in data.columns) else None

    # override if explicit facet columns are provided
    if facet_row_col and facet_row_col in data.columns:
        facet_row = facet_row_col
    if facet_col_col and facet_col_col in data.columns:
        facet_col = facet_col_col

    # ---- Category orders (safe) ----
    category_orders = {}
    if tag_col in data.columns and hasattr(data[tag_col], "cat"):
        category_orders[tag_col] = list(data[tag_col].cat.categories)
    if level_col and level_col in data.columns and hasattr(data[level_col], "cat"):
        category_orders[level_col] = list(data[level_col].cat.categories)

    # ---- Build figure ----
    fig = px.bar(
        data,
        x=tag_col,
        y=metric_col,
        color=(color_col if color_col in data.columns else None),
        barmode="group",
        text=(text_col if text_col in data.columns else None),
        facet_row=facet_row,
        facet_col=facet_col,
        category_orders=category_orders,
        labels={tag_col: "Entity Tag", metric_col: y_axis_label},
        title=title,
    )

    # ---- Text/axis formatting ----
    if text_col and text_col in data.columns:
        fig.update_traces(textposition="auto", insidetextanchor="middle", cliponaxis=False)
    else:
        fig.update_traces(
            texttemplate=f"%{{y:.{decimals}f}}",
            textposition="auto",
            insidetextanchor="middle",
            cliponaxis=False,
        )

    fig.update_yaxes(tickformat=f".{decimals}f", rangemode="tozero", title=y_axis_label)

    # ---- Layout ----
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=20, r=20, b=40),
        legend_title=None,
        bargap=0.25,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        height=(height if height else (700 if (facet_row or facet_col) else 600)),
    )
    fig.update_xaxes(tickangle=-30)

    # ---- Independent y-axis for facets ----
    if facet_row or facet_col:
        fig.for_each_yaxis(lambda yaxis: yaxis.update(matches=None, autorange=True))

    return html.Div([dcc.Graph(figure=fig)])


def plot_confidence_heatmaps_px(
    df: pd.DataFrame,
    *,
    title: str = "Confidence Confusion Heatmap (Mean Confidence on Errors)",
    panel_by: str = "Language",            # facet column
    value_col: str = "Overlap Count",      # mean or sum produced by your helper
    tag_order: list[str] | None = None,    # enforce axis order
    colorscale: str = "RdBu_r",
    value_precision: int = 2,
    shared_scale: bool = False,
) -> html.Div:
    needed = {"Tag1", "Tag2", value_col}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        raise KeyError(f"Missing required columns: {missing}")

    d = df.copy()

    # Facet fallback if the requested column is missing
    if panel_by not in d.columns:
        d[panel_by] = "All"

    # Enforce category order (avoids random alpha order)
    if tag_order is None:
        tags = sorted(set(d["Tag1"]).union(d["Tag2"]))
    else:
        present = set(d["Tag1"]).union(d["Tag2"])
        tags = [t for t in tag_order if t in present]

    d["True Entity Tag"] = pd.Categorical(d["Tag1"], categories=tags, ordered=True)
    d["Predicted Entity Tag"] = pd.Categorical(d["Tag2"], categories=tags, ordered=True)

    # Shared color scale across facets (optional but nice)
    range_color = None
    if shared_scale:
        vals = d[value_col].astype(float).to_numpy()
        vals = vals[~np.isnan(vals)]
        if vals.size:
            range_color = [float(np.nanmin(vals)), float(np.nanmax(vals))]
        else:
            range_color = [0.0, 0.0]

    # Simple PX heatmap (uses your pre-aggregated values)
    fig = px.density_heatmap(
        d,
        x="Predicted Entity Tag",
        y="True Entity Tag",
        z=value_col,                 # we already aggregated; PX will just place values
        facet_col=panel_by,          # one panel per language
        color_continuous_scale=colorscale,
        category_orders={"True Entity Tag": tags, "Predicted Entity Tag": tags},
        range_color=range_color,
        title=title,
    )

    # Show the numeric value in each cell
    fig.update_traces(
        texttemplate=f"%{{z:.{value_precision}f}}",
        textfont=dict(size=12),
        hovertemplate="<b>True</b>: %{y}<br><b>Pred</b>: %{x}<br>"
                      + f"<b>Value</b>: %{{z:.{value_precision}f}}<extra></extra>"
    )

    # Cosmetics
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title="Total Confidence"),
        margin=dict(t=60, l=30, r=30, b=30),
        height=800,
        width=max(650, 800 * 2),  # scale with #panels
    )
    # Tilt x tick labels for readability
    for ax in fig.layout:
        if isinstance(ax, str) and ax.startswith("xaxis"):
            fig.layout[ax].update(tickangle=-45, showgrid=False)
        if isinstance(ax, str) and ax.startswith("yaxis"):
            fig.layout[ax].update(showgrid=False)

    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])


def render_eval_overall_table(df: pd.DataFrame, digits: int = 4) -> html.Div:
    disp = df.copy()

    def _fmt(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        return s.map(lambda x: f"{float(x):.{digits}f}" if pd.notna(x) else "")

    for idx in disp.index:
        disp.loc[idx] = _fmt(disp.loc[idx])

    def _cid(c0, c1): return f"{c0}|{c1}"

    columns = [{"name": ["", "Metric"], "id": "Metric"}]
    for c0, c1 in disp.columns:
        columns.append({"name": [str(c0), str(c1)], "id": _cid(c0, c1)})

    rows = []
    for metric in disp.index:
        row = {"Metric": metric}
        for c0, c1 in disp.columns:
            row[_cid(c0, c1)] = disp.loc[metric, (c0, c1)]
        rows.append(row)

    table = DataTable(
        id="overall-token-vs-entity-table",
        columns=columns,
        data=rows,
        merge_duplicate_headers=True,
        style_as_list_view=True,
        style_table={"width": "100%"},
        style_cell={"textAlign": "center", "padding": "10px"},
        style_header={
            "text-align": "center",
            "background-color": HEADER_BG,
            "color": "white",
            "fontWeight": "700",
        },
        style_data_conditional=[
            {"if": {"column_id": "Metric"}, "textAlign": "left", "fontWeight": "600"},
        ],
    )
    return html.Div([table])



def plot_tag_metric_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str = "Count",          # "Count" (raw) or "Scale" (proportion 0..1)
    title: str = "Error Breakdown by Entity Span (TP / FP / FN)",
    facet_row: str | None = "Language",
    facet_col: str | None = "Scheme",
    tag_order: list[str] | None = None,
    metric_order: list[str] | None = ("TP","FP","FN"),
    colorscale: str = "RdBu_r",
    value_precision: int = 0,          # use 2 for Scale proportions
    height: int = 800,
    width: int = 1200,
) -> html.Div:
    need = {"Tag","Metric", value_col}
    if not need.issubset(df.columns):
        missing = ", ".join(sorted(need - set(df.columns)))
        raise KeyError(f"Missing columns for heatmap: {missing}")

    d = df.copy()

    # Enforce axis orders
    if tag_order:
        present = [t for t in tag_order if t in d["Tag"].astype(str).unique().tolist()]
        if present:
            d["Tag"] = pd.Categorical(d["Tag"].astype(str), categories=present, ordered=True)
    if metric_order:
        present_m = [m for m in metric_order if m in d["Metric"].astype(str).unique().tolist()]
        if present_m:
            d["Metric"] = pd.Categorical(d["Metric"].astype(str), categories=present_m, ordered=True)

    # build heatmap with text labels
    fig = px.density_heatmap(
        d,
        x="Tag",
        y="Metric",
        z=value_col,
        facet_row=facet_row if facet_row in d.columns else None,
        facet_col=facet_col if facet_col in d.columns else None,
        color_continuous_scale=colorscale,
        category_orders={
            "Tag": list(d["Tag"].cat.categories) if hasattr(d["Tag"], "cat") else None,
            "Metric": list(d["Metric"].cat.categories) if hasattr(d["Metric"], "cat") else None,
        },
        title=title,
    )

    # show numbers in each cell
    is_prop = pd.api.types.is_numeric_dtype(d[value_col]) and d[value_col].dropna().between(0,1).all()
    precision = (2 if is_prop and value_precision == 0 else value_precision)
    fig.update_traces(
        texttemplate=f"%{{z:.{precision}f}}",
        textfont=dict(size=12),
        hovertemplate="<b>Metric</b>: %{y}<br><b>Tag</b>: %{x}<br><b>Value</b>: %{z:.3f}<extra></extra>",
    )

    # Shared color range (prevents each facet auto-scaling)
    vals = d[value_col].astype(float).to_numpy()
    vals = vals[~np.isnan(vals)]
    if vals.size:
        fig.update_layout(coloraxis=dict(cmin=float(np.nanmin(vals)), cmax=float(np.nanmax(vals))))

    # Cosmetics
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title=("Proportion" if is_prop else "Count")),
        margin=dict(t=60, l=30, r=30, b=30),
        height=height,
        width=width,
    )
    # tidy axes
    for ax in fig.layout:
        if isinstance(ax, str) and ax.startswith("xaxis"):
            fig.layout[ax].update(tickangle=-30, showgrid=False)
        if isinstance(ax, str) and ax.startswith("yaxis"):
            fig.layout[ax].update(showgrid=False)

    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])

# from dash import dcc, html
# import plotly.express as px
# import pandas as pd

# def plot_metric_bar(
#     df: pd.DataFrame,
#     metric_col: str,
#     title: str,
#     *,
#     color: str = None,
#     facet_row: str = None,
#     facet_col: str = None,
#     height: int = 800,
#     width: int = 1200,
#     text_round: int = 3
# ) -> html.Div:
#     """
#     Generic bar plot for evaluation metrics.

#     Parameters
#     ----------
#     df : DataFrame
#         Must contain `metric_col` and categorical columns for x, color, facets.
#     metric_col : str
#         Column name to plot on Y-axis (e.g. 'F1-score', 'Precision').
#     title : str
#         Figure title.
#     color : str, optional
#         Column name to color bars by.
#     facet_row : str, optional
#         Column name for row faceting.
#     facet_col : str, optional
#         Column name for column faceting.
#     height, width : int
#         Figure dimensions.
#     text_round : int
#         Decimal places to show on bar labels.
#     """

#     if "Tag" not in df.columns:
#         raise KeyError("Expected 'Tag' column in df for x-axis.")

#     fig = px.bar(
#         df,
#         x="Tag",
#         y=metric_col,
#         color=color,
#         facet_row=facet_row,
#         facet_col=facet_col,
#         barmode="group",
#         text=metric_col,
#         template="plotly_white",
#         title=title,
#         facet_col_spacing=0.15,
#     )

#     # Round text labels
#     fig.update_traces(
#         texttemplate=f"%{{text:.{text_round}f}}",
#         textposition="auto",
#         textangle=0    # 👈 force horizontal text

#     )

#     fig.update_layout(
#                     height=height, 
#                     width=width,
#                     bargap=0.25,        # spacing between different tags
#                     # bargroupgap=0.15,   # spacing between IOB1 vs IOB2 within the same tag
#     )

#     return dcc.Graph(figure=fig)


# # def plot_metric_bar(
# #     df: pd.DataFrame,
# #     metric_col: str,
# #     title: str,
# #     *,
# #     color: str = None,
# #     facet_row: str = None,
# #     facet_col: str = None,
# #     height: int = 800,
# #     width: int = 1200,
# #     text_round: int = 3,
# #     text_col: str | None = None,     # NEW: override text labels
# #     facet_col_spacing: float = 0.14, # roomier columns
# #     bargap: float = 0.28,            # space between tag groups
# #     bargroupgap: float = 0.18        # space between TP/FP/FN within a tag
# # ) -> html.Div:
# #     if "Tag" not in df.columns:
# #         raise KeyError("Expected 'Tag' column in df for x-axis.")

# #     # default text = metric column; allow override
# #     text_arg = text_col if (text_col and text_col in df.columns) else metric_col

# #     fig = px.bar(
# #         df,
# #         x="Tag",
# #         y=metric_col,
# #         color=color,
# #         facet_row=facet_row,
# #         facet_col=facet_col,
# #         barmode="group",
# #         text=text_arg,
# #         template="plotly_white",
# #         title=title,
# #         facet_col_spacing=facet_col_spacing,
# #     )

# #     # decide label formatting: numeric text gets rounded; others printed raw
# #     if pd.api.types.is_numeric_dtype(df[text_arg]):
# #         fig.update_traces(texttemplate=f"%{{text:.{text_round}f}}", textposition="auto", textangle=0)
# #     else:
# #         fig.update_traces(textposition="auto", textangle=0)

# #     # If the metric looks like a proportion 0..1, use % ticks
# #     looks_like_prop = pd.api.types.is_numeric_dtype(df[metric_col]) and df[metric_col].dropna().between(0, 1).all()
# #     if looks_like_prop:
# #         fig.update_yaxes(tickformat=".0%")

# #     fig.update_layout(
# #         height=height, width=width,
# #         bargap=bargap,
# #         bargroupgap=bargroupgap,
# #         legend_title=None,
# #         margin=dict(t=60, l=20, r=20, b=40),
# #     )
# #     fig.update_xaxes(tickangle=-30)

# #     return dcc.Graph(figure=fig)





def plot_span_confusion_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str = "Count",            # or "Share"
    text_col: str | None = "Count",      # numbers shown in cells
    row_by: str = "Scheme",              # each row is a scheme (IOB1/IOB2)
    col_by: str = "Language",               # each column is a model (AraBERTv02/BERT)
    tag_order: list[str] | None = None,  # e.g., ["LOC", "MISC", "ORG", "PER"]
    metric_order: list[str] | None = None,  # e.g., ["FP","FN"]
    colorscale: str = "RdBu_r",
    title: str = "TP/FP/FN Breakdown by Entity Span",
    height: int = 700,
    width: int = 1000,
) -> html.Div:
    needed = {"Tag", "Metric", value_col, row_by, col_by}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    d = df.copy()

    # Default orders
    if tag_order is None:
        tag_order = sorted(d["Tag"].unique().tolist())
    if metric_order is None:
        metric_order = ["FP", "FN"]

    # Global z-range so all panels share the same color scale
    vals = pd.to_numeric(d[value_col], errors="coerce").to_numpy()
    vals = vals[~np.isnan(vals)]
    zmin = float(vals.min()) if vals.size else 0.0
    zmax = float(vals.max()) if vals.size else 1.0

    row_keys = list(pd.unique(d[row_by]))
    col_keys = list(pd.unique(d[col_by]))

    fig = make_subplots(
        rows=len(row_keys),
        cols=len(col_keys),
        subplot_titles=[f"{c} - {r}" for r in row_keys for c in col_keys],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
        shared_yaxes=True,
        shared_xaxes=False,
    )

    for i, rkey in enumerate(row_keys, start=1):
        for j, ckey in enumerate(col_keys, start=1):
            sub = d[(d[row_by] == rkey) & (d[col_by] == ckey)]
            # pivot to Metric × Tag
            mat = (
                sub.pivot_table(index="Metric", columns="Tag", values=value_col, aggfunc="sum", observed=True )
                .reindex(index=metric_order, columns=tag_order)
                .fillna(0.0)
            )
            if text_col and text_col in sub.columns:
                txt = (
                    sub.pivot_table(index="Metric", columns="Tag", values=text_col, aggfunc="sum", observed=True )
                    .reindex(index=metric_order, columns=tag_order)
                    .fillna(0)
                )
                # cast to int if counts
                if text_col.lower() == "count":
                    txt = txt.astype(int)
            else:
                txt = None

            fig.add_trace(
                go.Heatmap(
                    z=mat.values,
                    x=mat.columns,
                    y=mat.index,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    coloraxis="coloraxis",
                    text=None if txt is None else txt.values,
                    texttemplate="%{text}" if txt is not None else None,
                    hovertemplate="Metric: %{y}<br>Tag: %{x}<br>Value: %{z}<extra></extra>",
                    showscale=False,  # single shared colorbar
                ),
                row=i, col=j
            )

    fig.update_layout(
        coloraxis=dict(colorscale=colorscale, cmin=zmin, cmax=zmax, colorbar=dict(title="Counts" if value_col=="Count" else "Share")),
        title=title,
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(t=80, l=20, r=20, b=20),
    )

    # cosmetics
    for ax in fig.layout:
        if isinstance(ax, str) and ax.startswith("xaxis"):
            fig.layout[ax].update(tickangle=-30)
    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])


def plot_metric_bar(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    *,
    x_col: str = "Tag",     
    x_axis_title="Entity Span",
    y_axis_title=None,
    color: str = None,
    facet_row: str = None,
    facet_col: str = None,
    height: int = 800,
    width: int = 1200,
    text_round: int = 3,
    text_col: str | None = None,     # overrides bar labels if provided
    facet_col_spacing: float = 0.14, # roomier columns
    bargap: float = 0.28,            # space between tag groups
    bargroupgap: float = 0.18,       # space between colors within a tag
    tag_order: list[str] | None = None,  # << control entity span order (e.g. ["LOC","MISC","ORG","PER"])
    color_map: dict | None = None,     # << NEW: user-defined colors
    y_as_percent: bool | None = None,   # <— NEW: force %, force decimals, or auto
    text_position: str = "auto",          # "inside" | "outside" | "auto"
) -> html.Div:
    if x_col not in df.columns:
        raise KeyError(f"Expected '{x_col}' column in df for x-axis.")
    
    data = df.copy()

    # --- enforce Tag order if provided ---
    category_orders = {}
    if tag_order:
        # keep only categories that actually appear, but preserve requested order
        cats = [t for t in tag_order if t in data[x_col].astype(str).unique().tolist()]
        if cats:
            data[x_col] = pd.Categorical(data[x_col].astype(str), categories=cats, ordered=True)
            category_orders[x_col] = cats

    # default text = metric column; allow override
    text_arg = text_col if (text_col and text_col in data.columns) else metric_col

    fig = px.bar(
        data,
        x=x_col,
        y=metric_col,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        barmode="group",
        text=text_arg,
        template="plotly_white",
        title=title,
        facet_col_spacing=facet_col_spacing,
        category_orders=(category_orders or None),
        color_discrete_map=color_map,   # << hook in the custom color map
    )
    # numeric vs non-numeric text labels
    if pd.api.types.is_numeric_dtype(data[text_arg]):
        numeric_values = pd.to_numeric(data[text_arg], errors="coerce")
        is_all_int = numeric_values.dropna().eq(numeric_values.dropna().astype(int)).all()
        if is_all_int:
            fig.update_traces(
                texttemplate="%{text:d}",   # integers (counts)
                textposition=text_position,
                textangle=0,
                cliponaxis=False
            )
        else:
            fig.update_traces(
                texttemplate=f"%{{text:.{text_round}f}}",  # floats
                textposition=text_position,
                textangle=0,
                cliponaxis=False
            )

    else:
        fig.update_traces(textposition=text_position, textangle=0, cliponaxis=False)
    
    if x_axis_title:
        fig.update_xaxes(title_text=x_axis_title)

    if y_axis_title:
        fig.update_yaxes(title_text=y_axis_title)

    # # Decide percent vs decimal on Y axis
    # if y_as_percent is None:
    #     looks_like_prop = (
    #         pd.api.types.is_numeric_dtype(data[metric_col])
    #         and data[metric_col].dropna().between(0, 1).all()
    #     )
    # else:
    #     looks_like_prop = bool(y_as_percent)

    # if looks_like_prop:
    #     fig.update_yaxes(tickformat=".0%")

    # --- handle Y axis formatting explicitly ---
    if y_as_percent is True:
        fig.update_yaxes(tickformat=".0%")
    elif y_as_percent is False:
        fig.update_yaxes(tickformat=".2f")

    fig.update_layout(
        height=height, width=width,
        bargap=bargap,
        # bargroupgap=bargroupgap,
        legend_title=None,
        margin=dict(t=60, l=20, r=20, b=40),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    fig.update_xaxes(tickangle=-30)

    return dcc.Graph(figure=fig)



def plot_entity_errors_heatmap(
    df: pd.DataFrame,
    *,
    row_by: str = "Scheme",         # IOB1 / IOB2
    col_by: str = "Language",       # Arabic / English
    value_col: str = "Count",
    title: str,
    colorscale: str = "RdBu_r",
    tag_order: list[str] | None = None,   # e.g., ["LOC","MISC","ORG","PER"]
    height: int = 700,
    width: int = 1200,
) -> html.Div:
    required = {"True", "Pred", value_col, row_by, col_by}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {', '.join(sorted(missing))}")

    d = df.copy()

    # Determine consistent tag order
    if tag_order is None:
        tags = sorted(set(d["True"]).union(d["Pred"]))
    else:
        present = set(d["True"]).union(d["Pred"])
        tags = [t for t in tag_order if t in present]

    # Shared z-range for all facets
    vals = pd.to_numeric(d[value_col], errors="coerce")
    zmin = float(vals.min()) if vals.size else 0.0
    zmax = float(vals.max()) if vals.size else 1.0

    row_keys = list(pd.unique(d[row_by]))
    col_keys = list(pd.unique(d[col_by]))

    fig = make_subplots(
        rows=len(row_keys),
        cols=len(col_keys),
        subplot_titles=[f"{r} - {c}" for r in row_keys for c in col_keys],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
        shared_yaxes=True,
        shared_xaxes=True,
    )

    idx = 0
    for i, r in enumerate(row_keys, start=1):
        for j, c in enumerate(col_keys, start=1):
            sub = d[(d[row_by] == r) & (d[col_by] == c)]
            mat = (
                sub.pivot_table(index="Pred", columns="True", values=value_col, aggfunc="sum", observed=False)
                  .reindex(index=tags, columns=tags)
                  .fillna(0)
            )

            fig.add_trace(
                go.Heatmap(
                    z=mat.values,
                    x=mat.columns,
                    y=mat.index,
                    colorscale=colorscale,
                    zmin=zmin, zmax=zmax,
                    coloraxis="coloraxis",
                    text=mat.values.astype(int),
                    texttemplate="%{text}",
                    hovertemplate="True: %{x}<br>Pred: %{y}<br>Count: %{text}<extra></extra>",
                    showscale=False,
                ),
                row=i, col=j
            )
            idx += 1
            fig.update_xaxes(title_text="True Entity", row=i, col=j)
            fig.update_yaxes(title_text="Predicted Entity", row=i, col=j)

    fig.update_layout(
        coloraxis=dict(colorscale=colorscale, cmin=zmin, cmax=zmax, colorbar=dict(title="Error Count")),
        title=title,
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(t=80, l=40, r=40, b=40),
    )
    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])

def plot_token_confusion_heatmap(
    df: pd.DataFrame,
    *,
    panel_by: str = "Language",
    title: str = "Token-level Misclassification Heatmap (True × Pred)",
    tag_order: list[str] | None = None,
    colorscale: str = "RdBu_r",
    height: int = 600,
    width: int = 1200,
) -> html.Div:
    needed = {"True", "Pred", "Count", panel_by}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    d = df.copy()

    # axis order (optional)
    if tag_order:
        present = set(d["True"]).union(d["Pred"])
        cats = [t for t in tag_order if t in present]
    else:
        cats = sorted(set(d["True"]).union(d["Pred"]))

    d["True"] = pd.Categorical(d["True"], categories=cats, ordered=True)
    d["Pred"] = pd.Categorical(d["Pred"], categories=cats, ordered=True)

    # global color range
    vals = pd.to_numeric(d["Count"], errors="coerce").to_numpy()
    vals = vals[~np.isnan(vals)]
    zmin = float(vals.min()) if vals.size else 0.0
    zmax = float(vals.max()) if vals.size else 1.0

    panels = list(pd.unique(d[panel_by]))
    fig = make_subplots(
        rows=1, cols=len(panels),
        subplot_titles=panels,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for j, key in enumerate(panels, start=1):
        sub = d[d[panel_by] == key]
        mat = (
            sub.pivot_table(index="Pred", columns="True", values="Count", aggfunc="sum", observed=False)
              .reindex(index=cats, columns=cats)
              .fillna(0)
        )
        txt = mat.astype(int)

        fig.add_trace(
            go.Heatmap(
                z=mat.values,
                x=mat.columns,
                y=mat.index,
                colorscale=colorscale,
                zmin=zmin, zmax=zmax,
                coloraxis="coloraxis",
                text=txt.values,
                texttemplate="%{text}",
                hovertemplate="True: %{x}<br>Pred: %{y}<br>Count: %{text}<extra></extra>",
                showscale=False,
            ),
            row=1, col=j
        )
        fig.update_xaxes(title_text="True Labels", row=1, col=j)
    fig.update_yaxes(title_text="Predicted Labels", row=1, col=1)

    fig.update_layout(
        coloraxis=dict(colorscale=colorscale, cmin=zmin, cmax=zmax, colorbar=dict(title="Counts")),
        title=title,
        template="plotly_white",
        height=height, width=width,
        margin=dict(t=70, l=20, r=20, b=20),
    )
    # tilt x tick labels for readability
    for ax in fig.layout:
        if isinstance(ax, str) and ax.startswith("xaxis"):
            fig.layout[ax].update(tickangle=-30)

    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])


def plot_support_corr_heatmaps(
    df: pd.DataFrame,
    *,
    title: str = "Pearson & Spearman Correlations (Support vs Precision/Recall)",
    height: int = 800,
    width: int = 1000,
) -> html.Div:
    # Expect tidy df from the helper
    needed = {"Language","Split","Method","Metric","Corr"}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        raise KeyError(f"Missing columns: {missing}")

    # make 4 panels: Pearson-Train, Pearson-Test, Spearman-Train, Spearman-Test
    def _pivot(method: str, split: str) -> pd.DataFrame:
        sub = df[(df["Method"] == method) & (df["Split"] == split)]
        if sub.empty:
            # ensure 2 rows (Precision, Recall) × columns (Languages)
            return pd.DataFrame(index=["Precision","Recall"])
        return sub.pivot(index="Metric", columns="Language", values="Corr").reindex(index=["Precision","Recall"])

    p_train = _pivot("Pearson",  "Train")
    p_test  = _pivot("Pearson",  "Test")
    s_train = _pivot("Spearman", "Train")
    s_test  = _pivot("Spearman", "Test")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Pearson (Train)", "Pearson (Test)",
            "Spearman (Train)", "Spearman (Test)"
        ),
        horizontal_spacing=0.20,
        vertical_spacing=0.12,
    )

    panels = [(p_train, 1, 1), (p_test, 1, 2), (s_train, 2, 1), (s_test, 2, 2)]
    for mat, r, c in panels:
        z   = mat.values
        txt = mat.round(3).values
        fig.add_trace(
            go.Heatmap(
                z=z, x=mat.columns, y=mat.index,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=txt, texttemplate="%{text}",
                hovertemplate="Metric: %{y}<br>Language: %{x}<br>Correlation: %{z:.3f}<extra></extra>",
                coloraxis="coloraxis",
                showscale=False,
            ),
            row=r, col=c
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        width=width,
        coloraxis=dict(colorbar=dict(title="Correlation"), cmin=-1, cmax=1, colorscale="RdBu_r"),
        margin=dict(t=80, l=50, r=50, b=50),
    )
    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])


def plot_support_vs_metric_scatter(
    points_df: pd.DataFrame,
    means_df: pd.DataFrame,
    *,
    color_map: dict | None = None,
    height: int = 600,
    width: int = 1000,
    trendline: str | None = "ols",
    tag_order: list[str] | None = None,
) -> html.Div:
    # assume frames are already language-filtered
    d = points_df.copy()
    if d.empty:
        return html.Div("No data available.")

    if tag_order:
        present = [t for t in tag_order if t in d["Tag"].unique().tolist()]
        if present:
            d["Tag"] = pd.Categorical(d["Tag"], categories=present, ordered=True)

    fig = px.scatter(
        d, x="Support Value", y="Value", color="Tag",
        facet_col="Metric", facet_row="Split",
        trendline=trendline, facet_row_spacing=0.10, facet_col_spacing=0.05,
        hover_data=["Tag"], labels={"Value":"Metric Value", "Support Value":"Support"},
        template="plotly_white", color_discrete_map=color_map or {},
    )

    # overlay means (already one language)
    for row in means_df.itertuples(index=False, name=None):
        lang, metric, split, m_sup, m_met, s_sup, s_met, max_sup, min_sup, max_met, min_met, spread = row
        r = (["Train","Test"].index(split) + 1)
        c = (["Precision","Recall"].index(metric) + 1)
        fig.add_vline(x=m_sup, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=c)
        fig.add_hline(y=m_met, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=c)
        fig.add_annotation(
            x=0.04, y=0.06, xref="paper", yref="paper",
            text=(f"<b>Mean Support:</b> {m_sup:.1f}<br>"
                  f"<b>Support SD:</b> {0.0 if pd.isna(s_sup) else s_sup:.1f}<br>"
                  f"<b>Mean Metric:</b> {m_met:.3f}<br>"
                  f"<b>Metric SD:</b> {0.0 if pd.isna(s_met) else s_met:.3f}"),
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(size=10, color="gray"), row=r, col=c
        )

    fig.update_layout(width=width, height=height, margin=dict(t=70, l=50, r=30, b=50))
    return dcc.Graph(figure=fig)


def plot_spearman_rankdiff_bars(
    df: pd.DataFrame,
    *,
    tag_order: list[str] | None = None,
    height: int = 700,
    width: int = 1000,
    language: str,
) -> html.Div:
    d = df.copy()
    if d.empty:
        return html.Div("No data available")

    if tag_order:
        d["Tag"] = pd.Categorical(d["Tag"], categories=tag_order, ordered=True)
    x_col = "Tag"
    category_orders = {}
    if tag_order:
        # keep only categories that actually appear, but preserve requested order
        cats = [t for t in tag_order if t in d[x_col].astype(str).unique().tolist()]
        if cats:
            d[x_col] = pd.Categorical(d[x_col].astype(str), categories=cats, ordered=True)
            category_orders[x_col] = cats
    

    fig = px.bar(
        d,
        x="Tag",
        y="Squared Rank Difference",
        facet_col="Split",
        facet_row="Metric",
        text="Squared Rank Difference",
        facet_col_spacing=0.15,
        category_orders=category_orders,
        title=f"Squared Rank Differences of Entity Tags in {language}",
        labels={"Squared Rank Difference": "Squared Rank Diff.", "Tag": "Entity Tag"},
        template="plotly_white",
        barmode="group",
    )
    fig.update_traces(textposition="auto")
    fig.update_layout(height=height, width=width, margin=dict(t=60, l=40, r=40, b=40))
    return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])


# def plot_support_vs_metric_scatter(
#     points_df: pd.DataFrame,
#     means_df: pd.DataFrame,
#     *,
#     language: str,
#     color_map: dict | None = None,
#     height: int = 600,
#     width: int = 1000,
#     trendline: str | None = "ols",
#     tag_order: list[str] | None = None,
# ) -> html.Div:
#     # Filter to one language (you can call this twice for AR/EN like your registries)
#     d = points_df[points_df["Language"] == language].copy()
#     if d.empty:
#         return html.Div("No data for this language.")

#     # Order tags (optional)
#     if tag_order:
#         present = [t for t in tag_order if t in d["Tag"].unique().tolist()]
#         if present:
#             d["Tag"] = pd.Categorical(d["Tag"], categories=present, ordered=True)

#     fig = px.scatter(
#         d,
#         x="Support Value",
#         y="Value",
#         color="Tag",
#         facet_col="Metric",   # Precision vs Recall
#         facet_row="Split",    # Train vs Test
#         trendline=trendline,
#         facet_row_spacing=0.10,
#         facet_col_spacing=0.05,
#         hover_data=["Tag"],
#         labels={"Value":"Metric Value", "Support Value":"Support"},
#         template="plotly_white",
#         color_discrete_map=color_map or {},
#     )

#     # Overlay facet means as dashed crosshairs + small annotation
#     # NB: itertuples returns flat tuples; unpack accordingly.
#     for row in means_df[means_df["Language"] == language].itertuples(index=False, name=None):
#         # Unpack in the same order as created in helper
#         lang, metric, split, m_sup, m_met, s_sup, s_met, max_sup, min_sup, max_met, min_met, spread = row
#         r = (["Train","Test"].index(split) + 1)
#         c = (["Precision","Recall"].index(metric) + 1)

#         fig.add_vline(x=m_sup, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=c)
#         fig.add_hline(y=m_met, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=c)

#         fig.add_annotation(
#             x=0.04, y=0.06, xref="paper", yref="paper",
#             text=(f"<b>Mean Support:</b> {m_sup:.1f}<br>"
#                   f"<b>Support SD:</b> {0.0 if pd.isna(s_sup) else s_sup:.1f}<br>"
#                   f"<b>Mean Metric:</b> {m_met:.3f}<br>"
#                   f"<b>Metric SD:</b> {0.0 if pd.isna(s_met) else s_met:.3f}"),
#             showarrow=False, xanchor="left", yanchor="bottom",
#             font=dict(size=10, color="gray"), row=r, col=c
#         )

#     fig.update_layout(
#         width=width, height=height,
#         margin=dict(t=70, l=50, r=30, b=50),
#     )
#     # Optional: cleaner axes
#     # fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)

#     return dcc.Graph(figure=fig)


# def plot_spearman_rankdiff_bars(
#     df: pd.DataFrame,
#     *,
#     language: str,
#     tag_order: list[str] | None = None,
#     height: int = 700,
#     width: int = 1000,
# ) -> html.Div:
#     if df.empty:
#         return html.Div("No data available")

#     if tag_order:
#         df["Tag"] = pd.Categorical(df["Tag"], categories=tag_order, ordered=True)

#     fig = px.bar(
#         df[df["Language"] == language],
#         x="Tag",
#         y="Squared Rank Difference",
#         # color="Higher Column",
#         facet_col="Split",
#         facet_row="Metric",
#         text="Squared Rank Difference",
#         facet_col_spacing=0.15,
#         title=f"Squared Rank Differences of Entity Tags in {language}",
#         labels={"Squared Rank Difference": "Squared Rank Diff.", "Tag": "Entity Tag"},
#         template="plotly_white",
#         barmode="group",
#     )
#     fig.update_traces(textposition="auto")

#     fig.update_layout(
#         height=height,
#         width=width,
#         margin=dict(t=60, l=40, r=40, b=40),
#     )
#     return html.Div([dcc.Graph(figure=fig, config={"displaylogo": False})])


def ttr(df, height, width, tag_order=None):
    """
    Plots a faceted bar chart with datasets and splits displayed separately.
    """
    color_map = {
        "B-LOC": "darkgreen",
        "B-PERS": "deepskyblue",
        "B-PER": "deepskyblue",
        "B-ORG": "darkcyan",
        "B-MISC": "palevioletred",
        "I-LOC": "yellowgreen",
        "I-PERS": "lightblue",
        "I-PER": "lightblue",
        "I-ORG": "cyan",
        "I-MISC": "violet",
        "O": "saddlebrown",
    }

    category_orders = {}
    if tag_order:
        # make sure all expected tags are there, keep provided order
        present = [t for t in tag_order if t in df["Tag"].unique()]
        missing = [t for t in tag_order if t not in present]
        full_order = present + missing
        category_orders["Tag"] = full_order
        df = df.copy()
        df["Tag"] = pd.Categorical(df["Tag"], categories=full_order, ordered=True)
        

    fig = px.bar(
        df,
        x="Tag",
        y="TTR",
        color="Tag",
        facet_col="Language" if "Language" in df.columns else None,
        facet_row="Split" if "Split" in df.columns else None,
        color_discrete_map=color_map,
        text="TTR",
        title="Type-to-Token Ratio (TTR) Across Entity Tags in Arabic and English (Train and Test Splits)",
        labels={"Tag": "Entity Tag"},
        category_orders=category_orders or None,
    )

    # Update layout for better readability
    fig.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(t=60, l=20, r=20, b=20),
        # title_x=0.5,
    )

    # Update text formatting
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)

    return html.Div([dcc.Graph(figure=fig)])
