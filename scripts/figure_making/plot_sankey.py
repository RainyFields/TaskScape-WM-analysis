"""Sankey diagram of task → feature → condition trial flow for publication figures."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import plotly.graph_objects as go

workspace_root = Path(__file__).resolve().parents[3]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from figure_style.figure_utils import mm_to_inches

A4_WIDTH_MM = 210.0
FIG_SCALE = 0.7
BASE_WIDTH_MM = A4_WIDTH_MM / 2.0
BASE_HEIGHT_MM = BASE_WIDTH_MM * 3.0 / 2.0 * 0.6
FIG_WIDTH_MM = BASE_WIDTH_MM * FIG_SCALE
FIG_HEIGHT_MM = BASE_HEIGHT_MM * FIG_SCALE

# Column positions inside the Sankey domain (left | center | right)
COLUMN_X = (0.02, 0.50, 0.98)
SANKEY_DOMAIN = dict(x=[0.18, 0.82], y=[0.0, 1.0])

DEFAULT_SUBJECT = "sub-03"
DEFAULT_INPUT_ROOT = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/"
    "trial_level_betas/grouped_betas/task_relevant_only"
)
DEFAULT_OUTPUT_DIR = Path(
    "/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/sankey"
)

ALL_KNOWN_TASKS = [
    "dms_loc",
    "nback_loc",
    "nback_ctg",
    "nback_obj",
    "interdms_loc_ABAB",
    "interdms_loc_ABBA",
    "interdms_obj_ABAB",
    "interdms_obj_ABBA",
    "interdms_ctg_ABAB",
    "interdms_ctg_ABBA",
    "ctxdm_col",
    "ctxdm_lco",
]

TASK_COLORS = {
    "dms_loc": "#3775BA",
    "nback_loc": "#FFD700",
    "nback_ctg": "#F4EEAC",
    "nback_obj": "#FFF6CC",
    "interdms_loc_ABAB": "#9A4D8E",
    "interdms_loc_ABBA": "#D9B9D4",
    "interdms_obj_ABAB": "#7B3294",
    "interdms_obj_ABBA": "#D7C4E2",
    "interdms_ctg_ABAB": "#C2A5CF",
    "interdms_ctg_ABBA": "#EA84DD",
    "ctxdm_col": "#42949E",
    "ctxdm_lco": "#92E3F9",
}

TASK_LABELS = {
    "dms_loc": "DMS · location",
    "nback_loc": "n-back · location",
    "nback_ctg": "n-back · category",
    "nback_obj": "n-back · object",
    "interdms_loc_ABAB": "Inter-DMS · loc ABAB",
    "interdms_loc_ABBA": "Inter-DMS · loc ABBA",
    "interdms_obj_ABAB": "Inter-DMS · obj ABAB",
    "interdms_obj_ABBA": "Inter-DMS · obj ABBA",
    "interdms_ctg_ABAB": "Inter-DMS · ctg ABAB",
    "interdms_ctg_ABBA": "Inter-DMS · ctg ABBA",
    "ctxdm_col": "ctxDM · color",
    "ctxdm_lco": "ctxDM · LCO",
}

FEATURE_COLORS = {
    "Location": "#3775BA",
    "Object": "#42949E",
    "Category": "#9A4D8E",
}

FEATURE_ORDER = ["Location", "Object", "Category"]
DEFAULT_COLOR = "#CFCECE"
EXPORT_DPI = 300
FONT_FAMILY = "Arial, Helvetica, sans-serif"
# Nature-style minimums for side labels on a ~74 mm-wide panel
NODE_FONT_SIZE = 11
CONDITION_FONT_SIZE = 9
COLUMN_TITLE_FONT_SIZE = 11
TITLE_FONT_SIZE = 10
NODE_THICKNESS = 10
NODE_PAD = 5
CONDITION_LABEL_TARGET = 20
PLOT_Y_TOP = 0.94
PLOT_Y_BOTTOM = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot task-feature-condition Sankey diagram.")
    parser.add_argument("--subj", default=DEFAULT_SUBJECT, help="Subject ID, e.g. sub-06")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def get_task_color(task_name: str, alpha: float = 0.45) -> str:
    return hex_to_rgba(TASK_COLORS.get(task_name, DEFAULT_COLOR), alpha=alpha)


def format_condition_label(name: str, max_len: int = 28) -> str:
    if len(name) <= max_len:
        return name
    return f"{name[: max_len - 1]}…"


def sankey_x_to_paper(x_sankey: float) -> float:
    x0, x1 = SANKEY_DOMAIN["x"]
    return x0 + x_sankey * (x1 - x0)


def sankey_y_to_paper(y_sankey: float) -> float:
    y0, y1 = SANKEY_DOMAIN["y"]
    domain_y = y0 + y_sankey * (y1 - y0)
    return PLOT_Y_BOTTOM + (1.0 - domain_y) * (PLOT_Y_TOP - PLOT_Y_BOTTOM)


def value_weighted_y(order: list[str], weights: dict[str, float]) -> list[float]:
    total = sum(weights[item] for item in order)
    if total <= 0:
        return [0.5 for _ in order]

    centers: list[float] = []
    cursor = 0.01
    usable = 0.98
    for item in order:
        fraction = weights[item] / total
        height = fraction * usable
        centers.append(cursor + height / 2.0)
        cursor += height
    return centers


def parse_key_strict(key: str) -> tuple[str, str]:
    for task in sorted(ALL_KNOWN_TASKS, key=len, reverse=True):
        if key.startswith(task):
            suffix = key[len(task) :]
            if suffix.startswith("_"):
                suffix = suffix[1:]
            return task, suffix
    return "Unknown", key


def get_task_features(task_name: str) -> list[str]:
    task_lower = task_name.lower()
    if "ctxdm" in task_lower:
        return ["Location", "Object", "Category"]
    if "loc" in task_lower:
        return ["Location"]
    if "obj" in task_lower:
        return ["Object"]
    if "ctg" in task_lower or "cat" in task_lower:
        return ["Category"]
    return ["Unknown"]


def load_subject_data(pkl_path: Path) -> dict:
    with pkl_path.open("rb") as handle:
        return pickle.load(handle)


def build_node_layout(pkl_data: dict) -> tuple[list[str], list[str], list[str], dict[str, tuple[str, str]]]:
    unique_tasks: set[str] = set()
    unique_features: set[str] = set()
    condition_map: dict[str, tuple[str, str]] = {}

    for key in pkl_data:
        task_name, condition_suffix = parse_key_strict(key)
        unique_tasks.add(task_name)
        features = get_task_features(task_name)
        unique_features.update(features)
        condition_map[condition_suffix] = (features[0], task_name)

    sorted_tasks = [task for task in ALL_KNOWN_TASKS if task in unique_tasks]
    sorted_tasks.extend(sorted(task for task in unique_tasks if task not in sorted_tasks))

    sorted_features = [feature for feature in FEATURE_ORDER if feature in unique_features]
    sorted_features.extend(sorted(feature for feature in unique_features if feature not in sorted_features))

    sorted_conditions = sorted(
        condition_map.keys(),
        key=lambda name: (condition_map[name][0], condition_map[name][1], name),
    )
    return sorted_tasks, sorted_features, sorted_conditions, condition_map


def compute_node_weights(
    pkl_data: dict,
    sorted_tasks: list[str],
    sorted_features: list[str],
    sorted_conditions: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    task_weights = {task: 0.0 for task in sorted_tasks}
    feature_weights = {feature: 0.0 for feature in sorted_features}
    condition_weights = {condition: 0.0 for condition in sorted_conditions}

    for key, beta_list in pkl_data.items():
        count = float(len(beta_list))
        task_name, condition_suffix = parse_key_strict(key)
        if task_name in task_weights:
            task_weights[task_name] += count
        if condition_suffix in condition_weights:
            condition_weights[condition_suffix] += count
        for feature in get_task_features(task_name):
            if feature in feature_weights:
                feature_weights[feature] += count

    return task_weights, feature_weights, condition_weights


def add_side_labels(
    fig: go.Figure,
    labels: list[str],
    y_positions: list[float],
    side: str,
    fontsize: int,
) -> None:
    if side == "left":
        x_pos = 0.01
        xanchor = "left"
    else:
        x_pos = 0.99
        xanchor = "right"

    for label, y_pos in zip(labels, y_positions):
        if not label:
            continue
        fig.add_annotation(
            x=x_pos,
            y=sankey_y_to_paper(y_pos),
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor=xanchor,
            yanchor="middle",
            font=dict(family=FONT_FAMILY, size=fontsize, color="#1A1A1A"),
        )


def build_sankey_figure(pkl_data: dict, subject: str) -> go.Figure:
    sorted_tasks, sorted_features, sorted_conditions, condition_map = build_node_layout(pkl_data)
    task_x, feature_x, condition_x = COLUMN_X
    task_weights, feature_weights, condition_weights = compute_node_weights(
        pkl_data, sorted_tasks, sorted_features, sorted_conditions
    )

    task_y = value_weighted_y(sorted_tasks, task_weights)
    feature_y = value_weighted_y(sorted_features, feature_weights)
    condition_y = value_weighted_y(sorted_conditions, condition_weights)

    labels: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_colors: list[str] = []
    label_to_index: dict[str, int] = {}

    for idx, task in enumerate(sorted_tasks):
        label_to_index[task] = len(labels)
        labels.append("")
        node_x.append(task_x)
        node_y.append(task_y[idx])
        node_colors.append(TASK_COLORS.get(task, DEFAULT_COLOR))

    for idx, feature in enumerate(sorted_features):
        label_to_index[feature] = len(labels)
        labels.append("")
        node_x.append(feature_x)
        node_y.append(feature_y[idx])
        node_colors.append(FEATURE_COLORS.get(feature, DEFAULT_COLOR))

    condition_label_step = max(1, len(sorted_conditions) // CONDITION_LABEL_TARGET)
    condition_display_labels: list[str] = []
    for idx, condition in enumerate(sorted_conditions):
        label_to_index[condition] = len(labels)
        labels.append("")
        node_x.append(condition_x)
        node_y.append(condition_y[idx])
        _, task_name = condition_map[condition]
        node_colors.append(TASK_COLORS.get(task_name, DEFAULT_COLOR))
        if idx % condition_label_step == 0:
            condition_display_labels.append(format_condition_label(condition))
        else:
            condition_display_labels.append("")

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    for key, beta_list in pkl_data.items():
        count = len(beta_list)
        task_name, condition_suffix = parse_key_strict(key)
        if task_name not in label_to_index or condition_suffix not in label_to_index:
            continue

        flow_color = get_task_color(task_name, alpha=0.35)
        task_index = label_to_index[task_name]
        condition_index = label_to_index[condition_suffix]

        for feature in get_task_features(task_name):
            if feature not in label_to_index:
                continue
            feature_index = label_to_index[feature]

            sources.extend([task_index, feature_index])
            targets.extend([feature_index, condition_index])
            values.extend([count, count])
            link_colors.extend([flow_color, flow_color])

    width_px = int(mm_to_inches(FIG_WIDTH_MM) * EXPORT_DPI)
    height_px = int(mm_to_inches(FIG_HEIGHT_MM) * EXPORT_DPI)

    node_hover = (
        [TASK_LABELS.get(task, task) for task in sorted_tasks]
        + sorted_features
        + [format_condition_label(condition) for condition in sorted_conditions]
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                domain=SANKEY_DOMAIN,
                node=dict(
                    pad=NODE_PAD,
                    thickness=NODE_THICKNESS,
                    line=dict(color="white", width=0.6),
                    label=labels,
                    x=node_x,
                    y=node_y,
                    color=node_colors,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=node_hover,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="%{value} trials<extra></extra>",
                ),
            )
        ]
    )

    task_display = [TASK_LABELS.get(task, task) for task in sorted_tasks]
    add_side_labels(fig, task_display, task_y, side="left", fontsize=NODE_FONT_SIZE)
    add_side_labels(
        fig,
        condition_display_labels,
        condition_y,
        side="right",
        fontsize=CONDITION_FONT_SIZE,
    )

    for feature, y_pos in zip(sorted_features, feature_y):
        fig.add_annotation(
            x=sankey_x_to_paper(feature_x),
            y=sankey_y_to_paper(y_pos),
            xref="paper",
            yref="paper",
            text=feature,
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(family=FONT_FAMILY, size=NODE_FONT_SIZE, color="#1A1A1A"),
        )

    column_titles = [
        (task_x, "Task"),
        (feature_x, "Feature"),
        (condition_x, "Condition"),
    ]
    for x_pos, title in column_titles:
        fig.add_annotation(
            x=sankey_x_to_paper(x_pos),
            y=1.03,
            xref="paper",
            yref="paper",
            text=title,
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=COLUMN_TITLE_FONT_SIZE, color="#1A1A1A"),
            xanchor="center",
        )

    fig.update_layout(
        title=dict(
            text=subject.replace("sub-", "Subject "),
            x=0.0,
            xanchor="left",
            font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="#1A1A1A"),
        ),
        width=width_px,
        height=height_px,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=52, b=24),
    )
    return fig


def export_figure(fig: go.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        out_path = output_base.with_suffix(f".{suffix}")
        fig.write_image(str(out_path), scale=1)
        print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    pkl_path = args.input_root / f"{args.subj}_task_condition_betas.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing grouped beta pickle: {pkl_path}")

    pkl_data = load_subject_data(pkl_path)
    fig = build_sankey_figure(pkl_data, args.subj)
    output_base = args.output_dir / f"{args.subj}_sankey_diagram"
    export_figure(fig, output_base)


if __name__ == "__main__":
    main()
