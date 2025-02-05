#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair[all]",
#     "loguru",
#     "numpy",
#     "pandas",
# ]
# ///

# pyright: basic, reportUnknownMemberType=false, reportMissingTypeStubs=false

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
from loguru import logger

# A list of supported ORF sames to be used in membership checks.
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

# The labels for the two weeks to compare
COMPARISON_WEEKS = ("Week N - 1", "Week N")

# String literal constant to represent the possible output formats. Type checkers will
# enforce that the user select only from these options.
OUTPUT_FORMAT: Literal["html", "svg", "png", "pdf"] = "html"

# String literal constant to represent the supported altair themes. Type checkers will
# enforce that the user select only from these options.
ALTAIR_THEME: Literal[
    "default",
    "dark",
    "excel",
    "fivethirtyeight",
    "ggplot2",
    "googlecharts",
    "latimes",
    "powerbi",
    "quartz",
    "urbaninstitute",
    "vox",
] = "default"

# The height of the plot in pixels.
PLOT_HEIGHT = 600


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
) -> OrfDataset:
    """
    Parses a provided file path and extracts the ORF name, validating it is supported.

    Args:
        query_file (Path | str): Path to the input data file

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
    if len(SUPPORTED_ORFS) > 0:
        assert orf in SUPPORTED_ORFS, (
            f"The ORF, '{orf}', parsed from the file name '{query_file}' did not matched the provided list of expected ORFs."
        )

    # return the ORF information as a dataclass (this could also be a named tuple)
    return OrfDataset(orf, path)


def find_plotting_files(
    search_dir: Path | str,
) -> list[OrfDataset]:
    """
    Searches a specified directory for plotting files, filters the dataset, and converts each file into an OrfFile dataclass.

    Args:
        search_dir (Path | str): Directory path to search for plotting files

    Returns:
        list[OrfFile]: A list of parsed OrfFile dataclass objects containing file metadata
    """
    return [collect_orf_data(file) for file in Path(search_dir).glob("*plot.tsv.gz")]


def parse_plotting_files(
    orf_files: list[OrfDataset],
) -> list[OrfDataset]:
    """
    Reads a list of ORF data files and processes the data into numeric values.

    Args:
        orf_files (list[OrfFile]): List of OrfFile dataclass objects containing file metadata

    Returns:
        list[OrfFile]: The input OrfFile objects with their dataframes populated
    """
    # for each file, loop through and fill a parsed dataframe into the OrfDataset's
    # df field
    for file in orf_files:
        orf_df = pd.read_csv(
            file.path,
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["Amino Acid Substitution", "Associated Variants", "Week N - 1", "Week N"],
        )
        for triweek in COMPARISON_WEEKS:
            orf_df[triweek] = pd.to_numeric(orf_df[triweek])
            orf_df["y"] = np.where(orf_df[triweek] < 0.01, 0.01, orf_df[triweek])  # noqa: PLR2004
            orf_df[triweek] = orf_df["y"]
            orf_df[triweek] = pd.to_numeric(orf_df[triweek])

        file.df = orf_df

    return orf_files


def render_scatter_plot(
    orf_bundle: OrfDataset,
) -> OrfDataset:
    """
    Generates an interactive scatter plot for a given ORF dataset.

    Args:
        orf_bundle (OrfFile): A dataclass containing the ORF data and metadata

    Returns:
        OrfFile: The input OrfFile object with its chart attribute populated

    Raises:
        AssertionError: If the orf_bundle's dataframe is None
    """
    # Use an assertion to check our assumption that this function will only ever be
    # called when the dataframe field of our OrfDataset instance has been filled
    assert orf_bundle.df is not None

    # set the altair theme using the constant above
    alt.theme.enable(ALTAIR_THEME)

    # render the scatterplot base
    scatter = (
        alt.Chart(orf_bundle.df)
        .mark_circle(size=120)
        .encode(
            x=alt.X(
                f"{COMPARISON_WEEKS[0]}:Q",
                scale=alt.Scale(type="log", domain=[0.01, 1]),
                title=COMPARISON_WEEKS[0],
            ),
            y=alt.Y(
                f"{COMPARISON_WEEKS[1]}:Q",
                scale=alt.Scale(type="log", domain=[0.01, 1]),
                title=COMPARISON_WEEKS[1],
            ),
            color=alt.Color("Amino Acid Substitution:N"),
            tooltip=["Amino Acid Substitution", "Associated Variants"],
        )
    )

    # render the diagonal line
    line_data = pd.DataFrame(
        {
            COMPARISON_WEEKS[0]: [0.01, 1],
            COMPARISON_WEEKS[1]: [0.01, 1],
        },
    )
    line = (
        alt.Chart(line_data)
        .mark_line(
            strokeDash=[5, 5],  # This sets a dashed style
            color="black",
        )
        .encode(
            x=alt.X(f"{COMPARISON_WEEKS[0]}:Q", scale=alt.Scale(type="log", domain=[0.01, 1])),
            y=alt.Y(f"{COMPARISON_WEEKS[1]}:Q", scale=alt.Scale(type="log", domain=[0.01, 1])),
        )
    )

    # Combine the scatter plot and the line into one chart
    combined_chart = scatter + line

    # add the interactive chart to the Orf bundle and return
    orf_bundle.chart = (
        combined_chart.interactive()
        .properties(
            width="container",
            height=PLOT_HEIGHT,
        )
        .configure_axis(labelFontSize=14, titleFontSize=14)
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
    # find all the files available for plotting and collect them into a list of OrfDataset objects
    plotting_files = find_plotting_files(search_dir)

    # parse the file for each orf dataset into dataframes
    orf_data = parse_plotting_files(plotting_files)

    # use the parsed dataframes wrapped in OrfDataset objects to render each plot, wrapping that in
    # OrfDataset as well
    final_data_bundles = [render_scatter_plot(orf_bundle) for orf_bundle in orf_data]

    # for each fine dataset, write out the rendered plot in the requested format
    for orf_dataset in final_data_bundles:
        if orf_dataset.chart is None:
            continue

        if OUTPUT_FORMAT == "html":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.html")
        elif OUTPUT_FORMAT == "png":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.png")
        elif OUTPUT_FORMAT == "svg":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.svg")
        elif OUTPUT_FORMAT == "pdf":
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.pdf")
        else:
            # this branch should be unreachable because all the literals in the OUTPUT_FORMAT
            # constant have been covered.
            logger.warning("Unsupported output format requested. Defaulting to HTML.")
            orf_dataset.chart.save(f"{output_dir}/{orf_dataset.orf}.html")


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
