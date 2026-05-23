"""Build reproducible VKR tables and figures from saved experiment summaries.

The script intentionally uses existing ``summary_metrics*.csv`` files under
``reports/figures/summary``. It does not train models or touch checkpoints.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "best_val_mse",
    "test_mse",
    "test_mae",
    "test_r2",
    "test_rmse",
    "test_phm_score",
    "inference_ms_per_sample",
    "inference_samples_per_sec",
]

EXPORT_COLUMNS = [
    "family",
    "profile",
    "final_fit_mode",
    "window_size",
    "temporal_type",
    "best_val_mse",
    "test_mse",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_phm_score",
    "inference_ms_per_sample",
    "inference_samples_per_sec",
    "artifact_suffix",
    "source_csv",
]

FAMILY_LABELS = {
    "v2_frozen": "v2 frozen",
    "v2_nonfrozen": "v2 nonfrozen",
    "v3": "v3",
    "v3_frozen": "v3 frozen",
    "v3_unfrozen": "v3 unfrozen",
    "v3_rnn": "v3_rnn",
    "v4_tcn": "v4_tcn",
    "v5_odd": "v5_odd",
}

FAMILY_COLORS = {
    "v2_frozen": "#90A4AE",
    "v2_nonfrozen": "#78909C",
    "v3": "#607D8B",
    "v3_frozen": "#546E7A",
    "v3_unfrozen": "#455A64",
    "v3_rnn": "#2196F3",
    "v4_tcn": "#FF9800",
    "v5_odd": "#4CAF50",
}

MODE_MARKERS = {
    "frozen": "o",
    "finetune": "^",
    "unknown": "s",
}

MODE_HATCHES = {
    "frozen": "",
    "finetune": "//",
    "unknown": "..",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate VKR-ready tables and figures from saved summary_metrics CSV files."
    )
    parser.add_argument("--root", type=Path, default=repo_root, help="Repository root.")
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Directory with summary_metrics CSV files. Defaults to reports/figures/summary.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/XLSX/Markdown tables. Defaults to reports/tables_for_vkr.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Output directory for PNG figures. Defaults to reports/figures/summary.",
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        help="Profile used for best-model tables and overview figures.",
    )
    parser.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="Do not write the XLSX workbook.",
    )
    return parser.parse_args()


def discover_summary_csvs(summary_dir: Path) -> list[Path]:
    return sorted(summary_dir.rglob("summary_metrics*.csv"))


def infer_family(source_csv: Path, summary_dir: Path) -> str:
    rel_parts = source_csv.relative_to(summary_dir).parts
    run_dir = rel_parts[0] if rel_parts else source_csv.parent.name

    if run_dir == "train_rul_hybrid_v3_rnn":
        return "v3_rnn"
    if run_dir == "train_rul_hybrid_v4_tcn":
        return "v4_tcn"
    if run_dir == "train_rul_hybrid_v5_odd":
        return "v5_odd"
    if run_dir == "train_three_models_2_frozen":
        return "v2_frozen"
    if run_dir == "train_three_models_2_0_nonfrozen":
        return "v2_nonfrozen"
    if run_dir == "train_three_models_3_frozen":
        return "v3_frozen"
    if run_dir == "train_three_models_3_unfrozen":
        return "v3_unfrozen"
    if run_dir == "train_three_models_3":
        return "v3"
    return run_dir


def infer_profile(frame: pd.DataFrame, source_csv: Path, summary_dir: Path) -> str:
    if "mode" in frame.columns and not frame["mode"].dropna().empty:
        return str(frame["mode"].dropna().iloc[0])

    rel_parts = source_csv.relative_to(summary_dir).parts
    if len(rel_parts) >= 2:
        return rel_parts[1]
    return "unknown"


def infer_final_fit_mode(row: pd.Series, source_csv: Path) -> str:
    existing = row.get("final_fit_mode")
    if pd.notna(existing) and str(existing).strip():
        return str(existing).strip()

    temporal_type = str(row.get("temporal_type", "")).lower()
    if temporal_type.endswith("_frozen"):
        return "frozen"
    if temporal_type.endswith("_finetune"):
        return "finetune"

    source_text = str(source_csv).lower()
    if "unfrozen" in source_text or "nonfrozen" in source_text:
        return "finetune"
    if "frozen" in source_text:
        return "frozen"
    if "train_rul_hybrid_v" in source_text:
        return "finetune"
    return "unknown"


def normalize_model_name(value: object) -> str:
    model = str(value)
    model = model.removesuffix("_frozen").removesuffix("_finetune")
    return model


def collect_metrics(summary_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for csv_path in discover_summary_csvs(summary_dir):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue

        frame = frame.copy()
        frame["family"] = infer_family(csv_path, summary_dir)
        frame["profile"] = infer_profile(frame, csv_path, summary_dir)
        frame["source_csv"] = str(csv_path.relative_to(summary_dir))

        if "artifact_suffix" not in frame.columns:
            frame["artifact_suffix"] = csv_path.stem

        frame["final_fit_mode"] = frame.apply(
            lambda row: infer_final_fit_mode(row, csv_path), axis=1
        )
        frame["model_name"] = frame["temporal_type"].map(normalize_model_name)

        for column in METRIC_COLUMNS + ["window_size"]:
            if column not in frame.columns:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=EXPORT_COLUMNS + ["model_name"])

    metrics = pd.concat(frames, ignore_index=True, sort=False)
    metrics = metrics.sort_values(
        ["profile", "test_mse", "family", "final_fit_mode"],
        na_position="last",
    ).reset_index(drop=True)
    return metrics


def select_best_by_family_mode(metrics: pd.DataFrame, profile: str) -> pd.DataFrame:
    candidates = metrics.loc[
        (metrics["profile"] == profile) & metrics["test_mse"].notna()
    ].copy()
    if candidates.empty:
        return candidates

    idx = candidates.groupby(["family", "final_fit_mode"], dropna=False)["test_mse"].idxmin()
    best = candidates.loc[idx].sort_values(["test_mse", "family"]).reset_index(drop=True)
    return best


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in METRIC_COLUMNS:
        if column in rounded.columns:
            rounded[column] = rounded[column].round(6)
    return rounded


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join([":---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(value: object, row_index: int, column_index: int) -> str:
    reference = f"{_xlsx_col_name(column_index)}{row_index}"
    if pd.isna(value):
        return f'<c r="{reference}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = escape(str(value), quote=False)
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(frame: pd.DataFrame) -> str:
    rows = [list(frame.columns)] + frame.astype(object).values.tolist()
    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = [
            _xlsx_cell(value, row_index=row_index, column_index=column_index)
            for column_index, value in enumerate(row, start=1)
        ]
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def _safe_sheet_name(name: str) -> str:
    for char in "[]:*?/\\":
        name = name.replace(char, "_")
    return name[:31] or "Sheet"


def _write_minimal_xlsx(tables: dict[str, pd.DataFrame], path: Path) -> None:
    sheet_names = [_safe_sheet_name(name) for name in tables]
    workbook_sheets = "".join(
        f'<sheet name="{escape(name, quote=True)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    workbook_rels = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheet_names) + 1)
    )

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for index in range(1, len(sheet_names) + 1)
            )
            + '</Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{workbook_sheets}</sheets>'
            '</workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{workbook_rels}'
            '</Relationships>',
        )
        for index, frame in enumerate(tables.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(frame))


def try_write_excel(tables: dict[str, pd.DataFrame], path: Path) -> bool:
    try:
        _write_minimal_xlsx(tables, path)
    except Exception as exc:
        print(f"[warn] XLSX export skipped: {exc}")
        return False
    return True


def export_tables(
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    tables_dir: Path,
    skip_xlsx: bool = False,
) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_export = round_metrics(metrics.reindex(columns=EXPORT_COLUMNS))
    best_export = round_metrics(best.reindex(columns=EXPORT_COLUMNS))

    outputs = [
        tables_dir / "vkr_model_metrics_all.csv",
        tables_dir / "vkr_best_models_by_family_mode.csv",
        tables_dir / "vkr_best_models_by_family_mode.md",
    ]

    metrics_export.to_csv(outputs[0], index=False)
    best_export.to_csv(outputs[1], index=False)
    write_markdown_table(best_export, outputs[2])

    if not skip_xlsx:
        xlsx_path = tables_dir / "vkr_model_metrics.xlsx"
        if try_write_excel(
            {
                "all_metrics": metrics_export,
                "best_by_family_mode": best_export,
            },
            xlsx_path,
        ):
            outputs.append(xlsx_path)

    return outputs


def plot_best_mse(best: pd.DataFrame, output_path: Path, profile: str) -> Path | None:
    if best.empty:
        return None

    plot_data = best.sort_values("test_mse", ascending=True).reset_index(drop=True)
    labels = [
        f"{FAMILY_LABELS.get(row.family, row.family)} | {row.model_name} | {row.final_fit_mode}"
        for row in plot_data.itertuples(index=False)
    ]
    y_pos = range(len(plot_data))

    fig, ax = plt.subplots(figsize=(16, 8), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f5")
    ax.grid(axis="x", color="white", linewidth=1.4, zorder=0)

    bars = ax.barh(
        y_pos,
        plot_data["test_mse"],
        color=[FAMILY_COLORS.get(family, "#607D8B") for family in plot_data["family"]],
        hatch=[MODE_HATCHES.get(mode, "..") for mode in plot_data["final_fit_mode"]],
        edgecolor="white",
        linewidth=0.8,
        height=0.72,
        zorder=3,
    )

    for bar, value in zip(bars, plot_data["test_mse"]):
        ax.text(
            value + 0.0002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10,
            color="#333333",
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE", fontsize=12)
    ax.set_title(
        "Лучшие RUL-конфигурации по семействам и режимам обучения",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    fig.suptitle(
        f"Профиль: {profile}; источник: reports/figures/summary/**/summary_metrics*.csv",
        fontsize=11,
        color="#555555",
        y=0.94,
    )

    family_handles = [
        mpatches.Patch(facecolor=color, label=FAMILY_LABELS.get(family, family), edgecolor="white")
        for family, color in FAMILY_COLORS.items()
        if family in set(plot_data["family"])
    ]
    mode_handles = [
        mpatches.Patch(facecolor="#888888", hatch=hatch, label=mode, edgecolor="white")
        for mode, hatch in MODE_HATCHES.items()
        if mode in set(plot_data["final_fit_mode"])
    ]
    ax.legend(handles=family_handles + mode_handles, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_accuracy_speed(metrics: pd.DataFrame, output_path: Path, profile: str) -> Path | None:
    plot_data = metrics.loc[
        (metrics["profile"] == profile)
        & metrics["test_mse"].notna()
        & metrics["inference_ms_per_sample"].notna()
        & (metrics["inference_ms_per_sample"] > 0)
    ].copy()
    if plot_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f5")
    ax.grid(color="white", linewidth=1.2, zorder=0)

    for row in plot_data.itertuples(index=False):
        color = FAMILY_COLORS.get(row.family, "#607D8B")
        marker = MODE_MARKERS.get(row.final_fit_mode, "s")
        ax.scatter(
            row.inference_ms_per_sample,
            row.test_mse,
            s=140,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            normalize_model_name(row.temporal_type),
            xy=(row.inference_ms_per_sample, row.test_mse),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Время инференса, мс / sample (log scale)", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title(
        "Компромисс точности и скорости RUL-моделей",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    fig.suptitle(
        f"Профиль: {profile}; ниже и левее — лучше",
        fontsize=11,
        color="#555555",
        y=0.94,
    )

    family_handles = [
        mpatches.Patch(facecolor=color, label=FAMILY_LABELS.get(family, family), edgecolor="white")
        for family, color in FAMILY_COLORS.items()
        if family in set(plot_data["family"])
    ]
    mode_handles = [
        mlines.Line2D(
            [],
            [],
            marker=marker,
            color="white",
            label=mode,
            markerfacecolor="#888888",
            markeredgecolor="white",
            markersize=10,
            linestyle="None",
        )
        for mode, marker in MODE_MARKERS.items()
        if mode in set(plot_data["final_fit_mode"])
    ]
    ax.legend(handles=family_handles + mode_handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_materials(
    root: Path,
    summary_dir: Path | None = None,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    profile: str = "balanced",
    skip_xlsx: bool = False,
) -> list[Path]:
    summary_dir = summary_dir or root / "reports" / "figures" / "summary"
    tables_dir = tables_dir or root / "reports" / "tables_for_vkr"
    figures_dir = figures_dir or root / "reports" / "figures" / "summary"

    metrics = collect_metrics(summary_dir)
    best = select_best_by_family_mode(metrics, profile)

    outputs = export_tables(metrics, best, tables_dir, skip_xlsx=skip_xlsx)

    figure_2_18 = figures_dir / "figure_2_18_vkr_best_models_by_family_mode.png"
    figure_2_19 = figures_dir / "figure_2_19_vkr_accuracy_speed_tradeoff.png"
    for figure in [
        plot_best_mse(best, figure_2_18, profile),
        plot_accuracy_speed(metrics, figure_2_19, profile),
    ]:
        if figure is not None:
            outputs.append(figure)

    return outputs


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outputs = build_materials(
        root=root,
        summary_dir=args.summary_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        profile=args.profile,
        skip_xlsx=args.skip_xlsx,
    )
    print("Generated VKR materials:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
