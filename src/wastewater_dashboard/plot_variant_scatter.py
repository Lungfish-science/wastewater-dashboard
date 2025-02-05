#!/usr/bin/env python3

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

SUPPORTED_ORFS = [
    "ORF1",
    "S",
    "ORF3a",
    "E",
    "M",
    "ORF6",
    "ORF7a",
    "ORF7b",
    "ORF8",
    "N",
    "ORF10",
]
TRIWEEKS = ("W1", "W2")
OUTPUT_FORMAT = "html"


@dataclass
class OrfFile:
    orf: str
    path: Path | str
    df: pd.DataFrame | None = None
    chart: alt.LayerChart | None = None


def collect_orf_data(
    query_file: Path | str,
    expected_orfs: list[str],
) -> OrfFile:
    # make sure the provided file name has the expected extension
    assert "plot.tsv.gz" in str(query_file), "Unsupported file name supplied."

    # parse out the filename as a Path, and use to to retrieve the ORF
    path = Path(query_file)
    orf = path.stem.replace(".plot.tsv", "")

    # check that the parsed ORF is one of the expected ORFs
    if len(expected_orfs) > 0:
        assert orf in expected_orfs, (
            f"The ORF, '{orf}', parsed from the file name '{query_file}' did not matched the provided list of expected ORFs."
        )

    # return the ORF information as a dataclass (this could also be a named tuple)
    return OrfFile(orf, path)


def find_plotting_files(search_dir: Path | str, supported_orfs: list[str]) -> list[OrfFile]:
    return [collect_orf_data(file, supported_orfs) for file in Path(search_dir).glob("*plot.tsv.gz")]


def parse_plotting_files(orf_files: list[OrfFile], triweeks: tuple[str, str]) -> list[OrfFile]:
    for file in orf_files:
        orf_df = pd.read_csv(
            file.path,
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["Change", "Vars", "W1", "W2"],
        )
        for triweek in triweeks:
            orf_df[triweek] = pd.to_numeric(orf_df[triweek])
            orf_df["y"] = np.where(orf_df[triweek] < 0.01, 0.01, orf_df[triweek])
            orf_df[triweek] = orf_df["y"]
            orf_df[triweek] = pd.to_numeric(orf_df[triweek])

        file.df = orf_df

    return orf_files


def render_scatter_plot(orf_bundle: OrfFile, triweeks: tuple[str, str]) -> OrfFile:
    assert orf_bundle.df is not None
    scatter = (
        alt.Chart(orf_bundle.df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                f"{triweeks[0]}:Q",
                scale=alt.Scale(type="log", domain=[0.01, 1]),
                title=triweeks[0],
            ),
            y=alt.Y(
                f"{triweeks[1]}:Q",
                scale=alt.Scale(type="log", domain=[0.01, 1]),
                title=triweeks[1],
            ),
            color=alt.Color("Change:N"),
            tooltip=["Change", "Vars"],
        )
    )
    line_data = pd.DataFrame(
        {
            triweeks[0]: [0.01, 1],
            triweeks[1]: [0.01, 1],
        },
    )

    line = (
        alt.Chart(line_data)
        .mark_line(
            strokeDash=[5, 5],  # This sets a dashed style
            color="black",
        )
        .encode(
            x=alt.X(f"{triweeks[0]}:Q", scale=alt.Scale(type="log", domain=[0.01, 1])),
            y=alt.Y(f"{triweeks[1]}:Q", scale=alt.Scale(type="log", domain=[0.01, 1])),
        )
    )

    # Combine the scatter plot and the line into one chart
    combined_chart = scatter + line

    # add the interactive chart to the Orf bundle and return
    orf_bundle.chart = combined_chart.interactive().properties(width="container", height=500)

    return orf_bundle


def render_all_plots(search_dir: Path | str, output_dir: str | Path) -> None:
    plotting_files = find_plotting_files(search_dir, SUPPORTED_ORFS)
    orf_data = parse_plotting_files(plotting_files, TRIWEEKS)
    final_data_bundles = [render_scatter_plot(orf_bundle, TRIWEEKS) for orf_bundle in orf_data]

    for orf_dataset in final_data_bundles:
        if orf_dataset.chart is None:
            continue

        if OUTPUT_FORMAT == "html":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.html")
        elif OUTPUT_FORMAT == "png":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.png")
        elif OUTPUT_FORMAT == "svg":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.svg")
        else:
            pass


def main() -> None:
    assert len(sys.argv) == 3, "Usage: plot_variant_scatter.py <SEARCH_DIR> <OUTPUT_DIR>"  # noqa: PLR2004
    search_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    assert search_dir.is_dir(), f"The provided path, '{search_dir}', does not exist."
    render_all_plots(search_dir, output_dir)


if __name__ == "__main__":
    main()
