#!/usr/bin/env python3

# pyright: basic, reportUnknownMemberType=false, reportMissingTypeStubs=false

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
OUTPUT_FORMAT: Literal["html", "svg", "png", "pdf"] = "html"


@dataclass
class OrfDataset:
    """
    A state-machine dataclass for containing metadata for each orf dataset alongside
    each ORF's dataframe and rendered altair chart.
    """

    orf: str
    path: Path | str
    df: pd.DataFrame | None = None
    chart: alt.LayerChart | None = None


def collect_orf_data(
    query_file: Path | str,
    expected_orfs: list[str],
) -> OrfDataset:
    """
    Parses a provided file path and extracts the ORF name, validating it is supported.

    Args:
        query_file (Path | str): Path to the input data file
        expected_orfs (list[str]): List of valid ORF names to validate against

    Returns:
        OrfFile: A dataclass containing the parsed ORF name and file path

    Raises:
        AssertionError: If filename format is invalid or ORF is not in expected list
    """
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
    return OrfDataset(orf, path)


def find_plotting_files(
    search_dir: Path | str,
    supported_orfs: list[str],
) -> list[OrfDataset]:
    """
    Searches a specified directory for plotting files, filters the dataset, and converts each file into an OrfFile dataclass.

    Args:
        search_dir (Path | str): Directory path to search for plotting files
        supported_orfs (list[str]): List of valid ORF names to validate against

    Returns:
        list[OrfFile]: A list of parsed OrfFile dataclass objects containing file metadata
    """
    return [collect_orf_data(file, supported_orfs) for file in Path(search_dir).glob("*plot.tsv.gz")]


def parse_plotting_files(
    orf_files: list[OrfDataset],
    triweeks: tuple[str, str],
) -> list[OrfDataset]:
    """
    Reads a list of ORF data files and processes the data into numeric values.

    Args:
        orf_files (list[OrfFile]): List of OrfFile dataclass objects containing file metadata
        triweeks (tuple[str, str]): Tuple containing the two triweek labels to process

    Returns:
        list[OrfFile]: The input OrfFile objects with their dataframes populated
    """
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
            orf_df["y"] = np.where(orf_df[triweek] < 0.01, 0.01, orf_df[triweek])  # noqa: PLR2004
            orf_df[triweek] = orf_df["y"]
            orf_df[triweek] = pd.to_numeric(orf_df[triweek])

        file.df = orf_df

    return orf_files


def render_scatter_plot(
    orf_bundle: OrfDataset,
    triweeks: tuple[str, str],
) -> OrfDataset:
    """
    Generates an interactive scatter plot for a given ORF dataset.

    Args:
        orf_bundle (OrfFile): A dataclass containing the ORF data and metadata
        triweeks (tuple[str, str]): Tuple containing the two triweek labels to plot

    Returns:
        OrfFile: The input OrfFile object with its chart attribute populated

    Raises:
        AssertionError: If the orf_bundle's dataframe is None
    """
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
    orf_bundle.chart = combined_chart.interactive().properties(
        width="container",
        height=500,
    )

    return orf_bundle


def render_all_plots(search_dir: Path | str, output_dir: str | Path) -> None:
    """
    Renders scatter plots for all valid ORF files in a specified directory and saves them as
    output files in a specified format.

    Args:
        search_dir (Path | str): Directory path to search for plotting files
        output_dir (str | Path): Directory path to save rendered plot files

    Returns:
        None
    """
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
    """
    Program entrypoint if run as an executable script
    """
    assert len(sys.argv) == 3, (  # noqa: PLR2004
        "Usage: plot_variant_scatter.py <SEARCH_DIR> <OUTPUT_DIR>"
    )
    search_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    assert search_dir.is_dir(), f"The provided path, '{search_dir}', does not exist."
    render_all_plots(search_dir, output_dir)


if __name__ == "__main__":
    main()
