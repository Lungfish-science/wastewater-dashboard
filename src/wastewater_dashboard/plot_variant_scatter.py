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

import datetime
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import altair as alt
import polars as pl
from loguru import logger
from typing_extensions import Self

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

# The expected major lineage
# TODO(@Nick): This will be replaced with a command line arg at some point
MAJOR_LINEAGES = [
    "XEC",
    "XEC.4",
    "KP.3.1.1",
    "MC.10.1",
    "PA.1",
    "LP.8.1",
    "LF.7",
    "LB.1.3.1",
    "XEK",
    "XEQ",
]


@dataclass
class TimeWindows:
    # This field is used only for initialization.
    _input_list: list[str]

    latest_window: str = field(init=False)
    previous_window: str = field(init=False)

    def __post_init__(self) -> Self | None:
        timespans: list[str] = []
        for entry in self._input_list:
            if "--" not in entry:
                continue
            timespans.append(entry)
        if len(timespans) != 2:  # noqa: PLR2004
            logger.error(
                f"The provided input list does contain the expected date-range information, e.g., '2024-12-15--2025-01-04': {self._input_list}",
            )
            return None

        # Unpack the input list into the actual fields
        first_dates = [
            datetime.datetime.strptime(timespan.split("--")[0], "%Y-%m-%d").astimezone(datetime.timezone.utc)
            for timespan in timespans
        ]
        if first_dates[0] < first_dates[1]:
            logger.debug(f"Setting the `previous_window` attribute to {timespans[0]}")
            self.previous_window = timespans[0]
            logger.debug(f"Setting the `latest_window` attribute to {timespans[1]}")
            self.latest_window = timespans[1]
        else:
            logger.debug(f"Setting the `previous_window` attribute to {timespans[1]}")
            self.previous_window = timespans[1]
            logger.debug(f"Setting the `latest_window` attribute to {timespans[0]}")
            self.latest_window = timespans[0]


@dataclass
class OrfDataset:
    """
    A state-machine dataclass for containing metadata for each orf dataset alongside
    each ORF's dataframe and rendered altair chart.
    """

    orf: str
    df: pl.DataFrame
    windows: TimeWindows
    chart: alt.LayerChart | None = None


def setup_logging(level: int = 0) -> None:
    """
    Configure logging settings using loguru library.

    Removes default handler and configures a new stderr handler with colorization and the requested
    logging level.

    Args:
        level (int): Determines logging level:
            0 = WARNING (default)
            1 = SUCCESS
            2 = INFO
            3 = DEBUG

    Returns:
        None
    """
    # determine the correct logging level based on the optionally provided integer,
    # defaulting to setting the level at warning.
    match level:
        case 0:
            level_str = "WARNING"
        case 1:
            level_str = "SUCCESS"
        case 2:
            level_str = "INFO"
        case 3:
            level_str = "DEBUG"
        case _:
            level_str = "WARNING"

    # get rid of the default logger loaded by loguru and replace it with a
    # colorized logger that outputs to standard error at the requested level.
    logger.remove()
    logger.add(sys.stderr, colorize=True, level=level_str)


def validate_date_columns(unchecked_df: pl.DataFrame) -> pl.LazyFrame:
    """
    Validate that all date columns in a polars DataFrame contain valid date information.

    Takes an unchecked polars DataFrame and performs two validations:
    1. Checks that date columns were parsed as expected date types
    2. Checks that start dates precede end dates

    Args:
        unchecked_df (pl.DataFrame): DataFrame containing date columns to validate

    Returns:
        pl.LazyFrame: Validated data with an added "Time Span" column
    """
    # get the types for the date columns to make sure everything was input and parsed correctly
    schema = unchecked_df.schema

    start_type = schema.get("Time Span Start")
    stop_type = schema.get("Time Span End")

    # make sure all the columns that need to be dates were actually parsed as dates
    assert start_type == pl.Date, (
        f"The column 'Time Span Start' could not properly be parsed as a date and was instead parsed as a {start_type}. Please double check that the input data in this column was in YYYY-MM-DD format."
    )
    assert stop_type == pl.Date, (
        f"The column 'Time Span End' could not properly be parsed as a date and was instead parsed as a {start_type}. Please double check that the input data in this column was in YYYY-MM-DD format."
    )

    # run a check to see if any start dates do not precede their respective end dates
    date_check = (
        unchecked_df.select("Time Span Start", "Time Span End")
        .with_columns(pl.col("Time Span Start").lt(pl.col("Time Span End")).alias("_date_check"))
        .select("_date_check")
        .to_series()
        .to_list()
    )

    # to make a more helpful error message, collect the indices of the invalid dates
    invalid_date_indices = [i for i, passing in enumerate(date_check) if not passing]

    # if there were no invalid dates, return a now-validated lazyframe
    if len(invalid_date_indices) == 0:
        return unchecked_df.lazy().with_columns(
            pl.concat_str(
                [
                    pl.col("Time Span Start").dt.to_string(),
                    pl.col("Time Span End").dt.to_string(),
                ],
                separator="--",
            ).alias("Time Span"),
        )

    # if the list of collected invalid indices isn't empty, use those indices to collect tuples of invalid
    # date pairs, and print those invalid date pairs in the assertion message.
    start_dates = unchecked_df.select("Time Span Start").to_series().to_list()
    end_dates = unchecked_df.select("Time Span End").to_series().to_list()
    invalid_dates = [
        (start_date, end_date)
        for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates, strict=True))
        if i in invalid_date_indices
    ]
    assert False not in date_check, (
        f"One or more start dates that do not precede the end dates were encountered:\n{invalid_dates}"
    )

    return unchecked_df.lazy()


def reduce_to_latest_window(multi_window_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Reduce a multi-window LazyFrame to the latest time window data.

    Takes a LazyFrame containing mutation data across multiple time windows and extracts
    only the data from the most recent two time windows.

    Args:
        multi_window_lf (pl.LazyFrame): LazyFrame containing mutation data across multiple time periods

    Returns:
        pl.LazyFrame: Filtered LazyFrame containing only the latest two time windows
    """
    pre_flattened_dates = (
        multi_window_lf.select("Time Span Start", "Time Span End")
        .unique()
        .top_k(2, by=["Time Span Start", "Time Span End"])
        .collect()
        .rows()
    )

    flattened_dates = []
    for dates in pre_flattened_dates:
        assert len(dates) == 2, (  # noqa: PLR2004
            f"Invalid state has been represented in the computed latest dates: {pre_flattened_dates}"
        )
        flattened_dates = [*flattened_dates, *dates]

    return multi_window_lf.filter(
        pl.col("Time Span Start").is_in(flattened_dates),
        pl.col("Time Span End").is_in(flattened_dates),
    )


def parse_plotting_file(orf_file: str | Path) -> list[OrfDataset]:
    """
    Parse a tab-separated file containing ORF abundance data and return a list of OrfDataset objects.

    This function reads a TSV file containing abundance data for different ORFs across two time windows.
    It extracts time window information from the header, processes the data using polars, filters for
    abundances above 2%, and partitions the data by ORF.

    Args:
        orf_file (str | Path): Path to the tab-separated input file containing ORF abundance data

    Returns:
        list[OrfDataset]: A list of OrfDataset objects, each containing:
            - orf: Name of the ORF
            - df: Polars DataFrame with abundance data
            - windows: TimeWindows object with the time periods
            - chart: Initially None, populated later with Altair chart
    """
    unchecked_df = pl.read_csv(orf_file, separator="\t", try_parse_dates=True)
    checked_lf = validate_date_columns(unchecked_df)
    reduced_lf = reduce_to_latest_window(checked_lf)

    # parse time window information and store it as a dataclass object
    latest_timespans = checked_lf.select("Time Span").unique().top_k(2, by=["Time Span"]).collect().rows()
    flattened_timespans = []
    for date in latest_timespans:
        assert len(date) == 1, f"Invalid state has been represented in the computed latest dates: {latest_timespans}"
        flattened_timespans.append(date[0])
    timespans = TimeWindows(flattened_timespans)

    # use the expected major lineages to generate a regex pattern for matching against each row's
    # associated lineages
    major_lineage_regex = "(" + "|".join(map(re.escape, MAJOR_LINEAGES)) + ")"

    # pull out the min and max date for the comparison
    min_date = timespans.previous_window.split("--")[0]
    max_date = timespans.latest_window.split("--")[-1]

    # extend the query plan
    all_orf_abundances = (
        reduced_lf.filter(pl.col("1k+ read samples") > 1)
        .select(
            "Position",
            "ORFs",
            "AA Change",
            "Associated Variants",
            "Count",
            "Abundance",
            "Time Span",
        )
        .collect()
        # TODO(@Nick): This may need to change to support more generic Abundance columns
        # that contain abundances for many time spans, which can then be used with the
        # altair selector to filter which points are displayed.
        .pivot(
            values="Abundance",
            index=["Position", "ORFs", "AA Change", "Associated Variants"],
            on="Time Span",
            aggregate_function="first",
        )
        .lazy()
        .rename(
            {
                "Associated Variants": "Associated Lineages",
                "ORFs": "ORF",
            },
        )
        # TODO(@Nick): This should be replaced with a Polars selector that finds all date-range
        # abundance columns and applies this filter to them.
        .filter(
            pl.any_horizontal(
                pl.col(timespans.previous_window),
                pl.col(timespans.latest_window),
            ).ge(0.02),
        )
        .with_columns(
            pl.col("Associated Lineages").str.extract_all(major_lineage_regex).list.join(",").alias("Major Lineages"),
            pl.lit(f"{min_date} to {max_date}").alias("Comparison"),
        )
    )

    # split off a dataframe copy for displaying all mutations across the whole genome
    whole_genome_lf = all_orf_abundances.with_columns(
        pl.concat_str([pl.col("ORF"), pl.col("AA Change")], separator=" ").alias(
            "AA Change",
        ),
        pl.lit("Whole Genome").alias("ORF"),
    )

    # execute the optimized query plan with `.collect()`, and then split out one dataframe
    # per ORF with `.partition_by("ORF")`
    orf_dfs = all_orf_abundances.collect().vstack(whole_genome_lf.collect()).partition_by("ORF", as_dict=True)

    # return a list of OrfDataset objects
    return [OrfDataset(orf=str(orf_label[0]), windows=timespans, df=orf_df) for orf_label, orf_df in orf_dfs.items()]


def render_diag_line(orf_bundle: OrfDataset) -> alt.Chart:
    """
    Create a diagonal trend line for a comparison scatter plot.

    This function generates a straight diagonal trend line representing the line of equality
    (y=x) for the scatter plot. Points below this line indicate a decrease in abundance
    between the two time windows, while points above indicate an increase.

    Args:
        orf_bundle (OrfDataset): Dataset containing the ORF data and time window information

    Returns:
        alt.Chart: An Altair chart object containing the rendered diagonal line
    """
    line_data = pl.DataFrame(
        {
            orf_bundle.windows.previous_window: [0.0001, 1],
            orf_bundle.windows.latest_window: [0.0001, 1],
        },
    )
    return (
        alt.Chart(line_data)
        .mark_line(
            strokeDash=[5, 5],  # This sets a dashed style
            color="black",
        )
        .encode(
            # TODO(@Nick): This will need to change, potentially to a more generic
            # "Previous Abundance" and "Latest Abundance" pair of columns, to support
            # browsing multiple timespan comparisons.
            x=alt.X(
                f"{orf_bundle.windows.previous_window}:Q",
                scale=alt.Scale(type="log", domain=[0.0001, 1]),
            ),
            y=alt.Y(
                f"{orf_bundle.windows.latest_window}:Q",
                scale=alt.Scale(type="log", domain=[0.0001, 1]),
            ),
        )
    )


def render_scatter_plot(orf_bundle: OrfDataset) -> OrfDataset:
    """
    Render a scatter plot of amino acid change frequencies between time windows for an ORF.

    Creates an interactive Altair scatter plot comparing variant frequencies between two time
    windows. The plot includes:
    - A diagonal trend line for reference
    - Selectable filtering by SARS-CoV-2 lineage
    - Interactive legend highlighting
    - Logarithmic scales
    - Tooltips with mutation and lineage details
    - Dynamic color-coding by amino acid change position

    Args:
        orf_bundle (OrfDataset): Dataset containing the ORF data and time window information

    Returns:
        OrfDataset: The input dataset with an added rendered Altair chart
    """
    # set the altair theme using the constant above
    alt.theme.enable(ALTAIR_THEME)

    # Create an interactive parameter for variant selection.
    # Replace the hard-coded list with a dynamic list if needed.
    lineage_param = alt.param(
        name="selected_variant",
        bind=alt.binding_select(
            options=["All", *MAJOR_LINEAGES],
            name="SARS-CoV-2 Lineage: ",
        ),
        value="All",  # Default value shows all points.
    )

    # Make a parameter that will render a dropdown for selecting the timespan comparison
    comparison_dropdown = alt.binding_select(
        options=orf_bundle.df.select("Comparison").unique().to_series().to_list(),
        name="Timespan Comparison: ",
    )
    timespan_selector = alt.selection_point(
        fields=["Comparison"],
        bind=comparison_dropdown,
        name="Timespan Comparison: ",
    )

    # Create an interaction parameter allowing the legend to be used to highlight
    # particular mutations in the plot
    aa_change_selection = alt.selection_point(fields=["AA Change"], bind="legend")

    # collect a list of the amino-acid change's nucleotide positions in order
    # (TODO<@Nick>: This will not work for multi-segment pathogens)
    ordered_aa_changes = (
        orf_bundle.df.sort("Position").select("AA Change").unique(maintain_order=True).to_series().to_list()
    )

    # construct a window box to interactively highlight portions of the plot
    highlight_box = alt.selection_interval()

    # render the scatterplot base
    scatter_chart = (
        alt.Chart(orf_bundle.df)
        .mark_circle(size=120)
        .encode(
            # TODO(@Nick): This will need to change, potentially to a more generic
            # "Previous Abundance" and "Latest Abundance" pair of columns, to support
            # browsing multiple timespan comparisons.
            x=alt.X(
                f"{orf_bundle.windows.previous_window}:Q",
                scale=alt.Scale(type="log", domain=[0.0001, 1]),
                title=orf_bundle.windows.previous_window,
            ),
            y=alt.Y(
                f"{orf_bundle.windows.latest_window}:Q",
                scale=alt.Scale(type="log", domain=[0.0001, 1]),
                title=orf_bundle.windows.latest_window,
            ),
            # Conditionally color points: if "All" is selected or if the selected variant
            # is found in the comma-separated "Associated Lineages", use the normal color;
            # otherwise, gray them out.
            color=alt.condition(
                "selected_variant == 'All' || indexof(split(datum['Major Lineages'], ','), selected_variant) >= 0",
                alt.Color(
                    "AA Change:N",
                    legend=alt.Legend(symbolLimit=1000, columns=2),
                    scale=alt.Scale(domain=ordered_aa_changes),
                ),
                alt.value("lightgray"),
            ),
            opacity=(alt.when(aa_change_selection).then(alt.value(1)).otherwise(alt.value(0.2))),
            # TODO(@Nick): This will need to change, potentially to a more generic
            # "Previous Abundance" and "Latest Abundance" pair of columns, to support
            # browsing multiple timespan comparisons.
            tooltip=[
                "AA Change",
                "Associated Lineages",
                orf_bundle.windows.latest_window,
                orf_bundle.windows.previous_window,
            ],
        )
        # Add the parameter to include the UI element in the chart.
        .add_params(lineage_param)
        # Add a parameter allowing interactive selections via the legend
        .add_params(aa_change_selection)
        # add the highlight box
        .add_params(highlight_box)
        # add the timespan comparison dropdown
        .add_params(timespan_selector)
        .transform_filter(timespan_selector)
    )

    # render the diagonal line
    line_chart = render_diag_line(orf_bundle)

    # Combine the scatter plot and the line into one chart
    combined_chart = scatter_chart + line_chart

    # set the chart to be interactive, make it auto-size to the user's screen width,
    # and make the text on the axes a little bigger
    orf_bundle.chart = combined_chart.properties(
        width="container",
        height=PLOT_HEIGHT,
    ).configure_axis(labelFontSize=14, titleFontSize=16)

    return orf_bundle


def write_rendered_plot(orf_dataset: OrfDataset, output_dir: str | Path) -> None:
    """
    Writes rendered plots to a specified output directory.

    Args:
        orf_dataset (OrfDataset): Dataset containing ORF data and rendered plot
        output_dir (str | Path): Directory path to save rendered plot files

    Returns:
        None
    """
    # skip to the next dataset if no chart has been generated
    if orf_dataset.chart is None:
        logger.warning(
            f"The scatter plot for {orf_dataset.orf} was missing and will be skipped. However, this may indicate that functions in this module are being run in an incorrect order.",
        )
        return

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
    # make sure the required arguments are provided
    assert len(sys.argv) == 3, (  # noqa: PLR2004
        "Usage: plot_variant_scatter.py <SEARCH_DIR> <OUTPUT_DIR>"
    )

    # set up the logger to use standard error at the warning level
    setup_logging(0)

    # parse the two positional arguments as input paths
    all_orf_tsv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # make sure the input path exists
    assert all_orf_tsv.is_file(), f"The provided path, '{all_orf_tsv}', does not exist."

    # parse the file for each orf dataset into dataframes
    orf_datasets = parse_plotting_file(all_orf_tsv)

    # use the parsed dataframes wrapped in OrfDataset objects to render each plot, wrapping that in
    # OrfDataset as well
    final_data_bundles = [render_scatter_plot(orf_dataset) for orf_dataset in orf_datasets]

    # for each fine dataset, write out the rendered plot in the requested format
    for orf_dataset in final_data_bundles:
        write_rendered_plot(orf_dataset, output_dir)


if __name__ == "__main__":
    main()
