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
PLOT_HEIGHT = 650

# The expected major lineage
# TODO(@Nick): This will be replaced with a command line arg at some point
LINEAGES_DF = pl.read_json("data/major_lineages.json")

MAJOR_LINEAGES = LINEAGES_DF["lineage"].to_list()


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
    chart: alt.LayerChart | None = None
    default_comparison: str | None = None
    sorted_comparisons: list[str] | None = None
    latest_group_idx: int | None = None

    def sort_comparisons(self) -> None:
        assert len(self.df) > 0, f"Empty dataframe encountered for {self.orf}, which cannot be sorted:\n\n{self.df}"
        # parse the latest date in a comparison back out of the comparison strings, which
        # are guaranteed to be valid because of how they were constructed and validated earlier
        # in the program
        sorted_comparisons_df = (
            self.df.select("Comparison", "Grouping")
            .unique(maintain_order=True)
            .with_columns(
                pl.col("Comparison")
                .str.split(" versus ")
                .list.first()
                .str.split(" to ")
                .list.last()
                .str.to_date()
                .alias("Most Recent End"),
            )
            .sort("Most Recent End", descending=True)
        )
        sorted_comparisons = sorted_comparisons_df.select("Comparison").to_series().to_list()
        default = sorted_comparisons[0]
        latest_group_idx = sorted_comparisons_df.filter(pl.col("Comparison").eq(default)).select("Grouping").item()

        # pull out the comparison strings sorted in descending order, and set the most recent
        # one to be the default displayed in the dropdown
        self.sorted_comparisons = sorted_comparisons
        self.default_comparison = default
        self.latest_group_idx = latest_group_idx

    def render_scatter_plot(self) -> None:
        """
        Render an interactive altair scatter plot using the data in this dataset.

        This plot combines multiple interactivity features:
        - A dropdown to select different timespan comparisons
        - A dropdown to filter points by SARS-CoV-2 lineage
        - Interactive highlighting via legend selection
        - Interactive click selection to track mutations across all timespans
        - Tooltips on hover showing detailed point metadata

        The plot shows abundance values in log scale from 0.0001 to 1.0, with points
        representing amino acid changes. Points above the diagonal line indicate
        increasing abundance between timespans, while points below indicate decreasing
        abundance. A legend allows identification of specific amino acid changes.

        When a mutation is clicked, all occurrences of that mutation across different
        timespans will be displayed, along with a connecting line showing its evolution,
        regardless of the selected timespan.

        Requires that `sort_comparisons()` has been called first to setup the comparison
        dropdown options.

        Returns:
            None: Modifies the class `chart` attribute in place
        """
        assert self.sorted_comparisons is not None, (
            "Be sure to run `self.sort_comparisons()` before calling this method."
        )
        assert self.default_comparison is not None, (
            "Be sure to run `self.sort_comparisons()` before calling this method."
        )

        # set the altair theme using the constant above
        alt.theme.enable(ALTAIR_THEME)

        # Create an interactive parameter for variant selection.
        lineage_param = alt.param(
            name="selected_variant",
            bind=alt.binding_select(
                options=["All", *MAJOR_LINEAGES],
                name="SARS-CoV-2 Lineage: ",
            ),
            value="All",  # Default value shows all points.
        )

        # Create an interaction parameter allowing the legend to be used to highlight
        # particular mutations in the plot
        aa_change_selection = alt.selection_point(fields=["AA Change"], bind="legend")

        # collect a list of the amino-acid change's nucleotide positions in order
        # (TODO<@Nick>: This will not work for multi-segment pathogens)
        ordered_aa_changes = (
            self.df.sort("Position").select("AA Change").unique(maintain_order=True).to_series().to_list()
        )

        # construct a window box to interactively highlight portions of the plot
        highlight_box = alt.selection_interval()

        # Make a parameter that will render a dropdown for selecting the timespan comparison
        comparison_dropdown = alt.binding_select(
            options=self.sorted_comparisons,
            name="Timespan Comparison: ",
        )
        _timespan_selector = alt.selection_point(
            fields=["Comparison"],
            bind=comparison_dropdown,
            name="Timespan Comparison: ",
            value=self.default_comparison,
        )

        # construct a click selector to allow users to click on each amino acid to see its path
        click_selection = alt.selection_point(
            fields=["AA Change"],
            on="click",
            empty=False,
            name="Click",
        )

        # X-value slider
        assert self.latest_group_idx is not None
        analysis_slider = alt.binding_range(
            min=1,
            max=self.latest_group_idx,
            step=1,
            name="Analysis Index: ",
        )
        analysis_selector = alt.selection_point(
            name="x_select",
            fields=["Grouping"],
            bind=analysis_slider,
            value=self.latest_group_idx,
        )

        # render the main scatterplot for the selected timespan
        scatter_chart = (
            alt.Chart(self.df)
            .mark_circle(size=120)
            .encode(
                x=alt.X(
                    "Abundance in Previous Time Span:Q",
                    scale=alt.Scale(type="log", domain=[0.001, 1]),
                    title="Abundance in Previous Time Span",
                ),
                y=alt.Y(
                    "Abundance in Current Time Span:Q",
                    scale=alt.Scale(type="log", domain=[0.001, 1]),
                    title="Abundance in Current Time Span",
                ),
                # Conditionally color points: if "All" is selected or if the selected variant
                # is found in the comma-separated "Associated Variants", use the normal color;
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
                opacity=alt.when(aa_change_selection).then(alt.value(1)).otherwise(alt.value(0.2)),
                tooltip=[
                    "AA Change",
                    "Associated Lineages",
                    "Major Lineages",
                    "Abundance in Current Time Span",
                    "Abundance in Previous Time Span",
                    "Comparison",
                    "ORF",
                ],
            )
            .add_params(aa_change_selection)
            .add_params(analysis_selector)
            .transform_filter(analysis_selector)
        )

        scatter_click = (
            alt.Chart(self.df)
            .mark_circle(size=200)
            .encode(
                x=alt.X(
                    "Abundance in Previous Time Span:Q",
                    scale=alt.Scale(type="log", domain=[0.001, 1]),
                ),
                y=alt.Y(
                    "Abundance in Current Time Span:Q",
                    scale=alt.Scale(type="log", domain=[0.001, 1]),
                ),
                color=alt.Color("AA Change:N"),
                opacity=alt.value(0.7),  # Show clicked mutations with consistent opacity
                tooltip=[
                    "AA Change",
                    "Associated Lineages",
                    "Major Lineages",
                    "Abundance in Current Time Span",
                    "Abundance in Previous Time Span",
                    "Comparison",
                    "ORF",
                ],
            )
            .transform_filter(click_selection)
        )  # Only filter by click, NOT by timespan

        # Line connecting occurrences of the clicked mutation - NOT filtered by timespan_selector
        click_line = (
            alt.Chart(self.df)
            .mark_line(size=2)
            .encode(
                x="Abundance in Previous Time Span:Q",
                y="Abundance in Current Time Span:Q",
                color=alt.value("black"),
                order=alt.Order("Grouping:Q", sort="ascending"),  # Ensures the line follows chronological order
                opacity=alt.value(0.5),
            )
            .transform_filter(click_selection)
        )  # Only filter by click, NOT by timespan

        # render the diagonal line
        line_chart = render_diag_line()

        # specify the comparison to be printed in the background
        background_comparison = (
            alt.Chart(self.df)
            .mark_text(x=0.001, y=1, dy=-20, align="left", fontSize=18, opacity=1)
            .encode(text="Comparison:N")
            .transform_filter(analysis_selector)
        )

        # Combine all charts with the correct layering and ALL parameters
        combined_chart = (
            alt.layer(
                background_comparison,
                scatter_chart,
                scatter_click,
                click_line,
                line_chart,
            )
            .configure_view(clip=False)
            .configure_axis(labelFontSize=14, titleFontSize=14)
            .properties(
                width="container",
                height=PLOT_HEIGHT,
                padding=10,
            )
            .add_params(
                lineage_param,
                click_selection,
                highlight_box,
            )
        )

        # Set final chart configuration
        self.chart = combined_chart.configure_axis(labelFontSize=14, titleFontSize=16)


@dataclass
class GroupNode:
    label: str
    left: int | None = None
    right: int | None = None


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
                separator=" to ",
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


def identify_timespan_pairs(checked_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create pairwise groupings from a time-ordered set of unique time spans.

    Takes a polars LazyFrame containing time spans and creates groups between pairs of windows
    for comparison. Each group consists of a 'previous' timespan and a 'current' timespan
    to compare mutation abundances between.

    Args:
        checked_lf (pl.LazyFrame): LazyFrame containing validated time span data

    Returns:
        pl.LazyFrame: LazyFrame with added Grouping and Comparison columns mapping pairs
            of time spans together
    """
    # peel off the unique time spans to be associated with each other
    unique_timespans = checked_lf.select("Time Span").unique(maintain_order=True).collect().to_series().to_list()

    # construct a sort of tree, which can be used to assign indices to each "forward"
    # and "reverse" pairing
    group_lookup: dict[str, GroupNode] = {span: GroupNode(span) for span in unique_timespans}
    group_index = 1
    window_size = 3
    for i, node in enumerate(group_lookup.values()):
        if i + window_size > (len(unique_timespans) - 1):
            assert node.right is None
            break

        next_pairing = unique_timespans[i + 3]
        next_node = group_lookup[next_pairing]

        assert next_node.left is None

        node.right = group_index
        next_node.left = group_index

        group_index += 1

    # turn the tree into a dataframe,
    both_groupings = pl.DataFrame(
        {
            "Time Span": group_lookup.keys(),
            "Grouping 1": [node.left for _, node in group_lookup.items()],
            "Grouping 2": [node.right for _, node in group_lookup.items()],
        },
    )

    # split into one per each of the two groupings, removing nulls representing rows where
    # a forward comparison isn't available yet, or when a time span is to early to be used
    # for reverse comparisons.
    grouping1 = (
        both_groupings.select("Time Span", "Grouping 1")
        .filter(pl.col("Grouping 1").is_not_null())
        .rename({"Grouping 1": "Grouping"})
    )
    grouping2 = (
        both_groupings.select("Time Span", "Grouping 2")
        .filter(pl.col("Grouping 2").is_not_null())
        .rename({"Grouping 2": "Grouping"})
    )

    # stack up the groups and create a column stating which time span is associated
    # with each group
    all_groups = (
        grouping1.vstack(grouping2)
        .lazy()
        .group_by("Grouping")
        .all()
        .with_columns(
            pl.when(pl.col("Time Span").list.len() == 2)  # noqa: PLR2004
            .then(pl.col("Time Span").list.join(" versus "))
            .otherwise(pl.lit(None))
            .alias("Comparison"),
        )
        .explode("Time Span")
    )

    # invert the group indices
    max_group = all_groups.select("Grouping").max().collect().item()
    all_groups = all_groups.with_columns(
        (pl.lit(max_group) - pl.col("Grouping") + 1).alias("Grouping"),
    )

    # return a lazyframe query for the input lazyframe joined to the groups dataframe
    return checked_lf.join(
        all_groups,
        on="Time Span",
        how="left",
        validate="m:m",
    )


def validate_pivot_groupings(lf_with_groupings: pl.LazyFrame) -> None:
    """
    Validate groupings before performing the pivot operation.

    This function checks that groups that will be constructed by the pivot operation
    contain no more than two rows (representing the two time spans being compared).
    If invalid groups are found (containing more than 2 rows), an assertion error
    is raised, as this indicates ambiguous or invalid input data that should not
    be plotted.

    Args:
        lf_with_groupings (pl.LazyFrame): LazyFrame containing data with grouping columns

    Raises:
        AssertionError: If any pivot groups contain more than 2 rows, indicating invalid data

    Returns:
        None
    """
    # partition the input lazyframe that contains groupings into a list of dataframes,
    # one per comparison.
    pivot_test = (
        lf_with_groupings.with_columns(pl.col("Time Span Start").min().over("Comparison").alias("_min_group_date"))
        .with_columns(
            pl.when(pl.col("Time Span Start").eq(pl.col("_min_group_date")))
            .then(pl.lit("Abundance in Previous Time Span"))
            .otherwise(pl.lit("Abundance in Latest Time Span"))
            .alias("Which Time Span"),
        )
        .drop("_min_group_date")
        .collect()
        .partition_by("Comparison")
    )

    # collect a list of any dataframes where a group that will be constructed by the pivot
    # operation will contain more than the maximum number of rows, 2
    max_rows_per_group = 2
    invalid_groups = []
    for big_groups in pivot_test:
        for df in big_groups.unique().partition_by(
            [
                "Position",
                "ORFs",
                "AA Change",
                "Associated Variants",
                "Comparison",
            ],
        ):
            if len(df) <= max_rows_per_group:
                continue
            invalid_groups.append(df)
    if len(invalid_groups) == 0:
        return

    # run an assertion that no groups are invalid. This should crash the program, as
    # it indicates ambiguous or invalid input data that should not be plotted.
    invalid_groups = [
        group.with_columns(pl.lit(f"Invalid Group {i + 1}").alias("Invalid Group Index"))
        for i, group in enumerate(invalid_groups)
    ]

    all_invalid_groups = pl.concat(invalid_groups)
    assert len(invalid_groups) == 0, (
        f"Pivot groups for this plot may only contain 1 or 2 rows, but the following groups contained more. This is likely because of duplicate entries of nucleotide/amino-acid substitutions in a given time span.\n\n{all_invalid_groups}"
    )


def transform_for_plotting(with_groupings_lf: pl.LazyFrame, major_lineage_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transform time-windowed mutation data for scatter plot visualization.

    Takes a LazyFrame containing mutation data with groupings between time spans, filters
    rows for sufficient read depth, constructs columns for abundance in previous and current
    time spans, and includes major lineage annotations.

    Args:
        with_groupings_lf (pl.LazyFrame): LazyFrame containing mutation data with time span
            groupings

    Returns:
        pl.lazyframe: Transformed LazyFrame with abundance columns and lineage annotations,
            filtered for plotting requirements
    """
    # make sure groupings based on the available information will work when put into
    # a pivot, which is to say, make sure that each grouping contains no more than 2
    # rows.
    validate_pivot_groupings(with_groupings_lf)

    # first, filter the input rows to make sure they contain enough reads, and then
    # create a column that states whether a row comes from the previous time span or the
    # current timespan. This column of two nominal values will be pivoted into the X-
    # and Y-columns on the scatter plot.
    filtered_long_df = (
        with_groupings_lf.unique()
        .filter(pl.col("1k+ read samples") > 1)
        .with_columns(
            pl.col("Time Span Start").min().over("Comparison").alias("_min_group_date"),
        )
        .with_columns(
            pl.when(pl.col("Time Span Start").eq(pl.col("_min_group_date")))
            .then(pl.lit("Abundance in Previous Time Span"))
            .otherwise(pl.lit("Abundance in Current Time Span"))
            .alias("Which Time Span"),
        )
        .drop("_min_group_date")
        .collect()
    )

    # run the aforementioned pivot, which, for each unique substitution in each time span
    # comparison, will create a column of abundances in the previous time span and the
    # current time span. Sometimes, a substitution will have dropped out (become frequency
    # 0 and thus be omitted from the input table), in which case one of these abundance
    # columns will be empty. These rows can be safely filtered out.
    pivot_lf = (
        filtered_long_df.pivot(
            values="Abundance",
            index=[
                "Position",
                "ORFs",
                "AA Change",
                "Associated Variants",
                "Grouping",
                "Comparison",
            ],
            on="Which Time Span",
        )
        .lazy()
        .filter(
            pl.col("Abundance in Previous Time Span").is_not_null(),
            pl.col("Abundance in Current Time Span").is_not_null(),
        )
    )

    agg_lineages = (
        major_lineage_lf.with_columns(
            pl.col("mutations").str.split(",").alias("_mutations"),
        )
        .explode("_mutations")
        .group_by("_mutations")
        .agg("lineage")
        .with_columns(pl.col("lineage").list.join(",").alias("lineage"))
        .with_columns(
            pl.col("_mutations")
            .str.split_exact(pl.lit(":"), 2)
            .struct.rename_fields(["ORFs", "AA Change"])
            .alias("_mutations"),
        )
        .unnest("_mutations")
        .rename({"lineage": "Major Lineages"})
    )
    with_major_lineages = pivot_lf.join(agg_lineages, how="left", on=["ORFs", "AA Change"])

    # return a lazyframe with a few final transformation: a filter to make sure at least
    # one of the abundance values is greater than 0.02, and some small column renames.
    return with_major_lineages.filter(
        pl.any_horizontal(
            pl.col("Abundance in Previous Time Span"),
            pl.col(
                "Abundance in Current Time Span",
            ).ge(0.02),
        ),
    ).rename({"ORFs": "ORF", "Associated Variants": "Associated Lineages"})


def parse_plotting_file(orf_file: str | Path, lineage_file: str | Path) -> list[OrfDataset]:
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
    lineage_lf = pl.read_json(lineage_file).lazy()
    checked_lf = validate_date_columns(unchecked_df)

    with_groupings = identify_timespan_pairs(checked_lf)
    all_orf_abundances = transform_for_plotting(with_groupings, lineage_lf)

    # write out plotting data for transparency
    all_orf_abundances.collect().write_csv("data/transformed_variant_plotting_data.tsv", separator="\t")

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
    assert all(len(orf_df) > 0 for _, orf_df in orf_dfs.items())

    # return a list of OrfDataset objects
    return [OrfDataset(orf=str(orf_label[0]), df=orf_df) for orf_label, orf_df in orf_dfs.items()]


def render_diag_line() -> alt.Chart:
    """
    Create a diagonal trend line for a comparison scatter plot.

    This function generates a straight diagonal trend line representing the line of equality
    (y=x) for the scatter plot. Points below this line indicate a decrease in abundance
    between the two time windows, while points above indicate an increase.

    Returns:
        alt.Chart: An Altair chart object containing the rendered diagonal line
    """
    line_data = pl.DataFrame(
        {
            "Abundance in Previous Time Span": [0.001, 1],
            "Abundance in Current Time Span": [0.001, 1],
        },
    )
    return (
        alt.Chart(line_data)
        .mark_line(
            strokeDash=[5, 5],  # This sets a dashed style
            color="black",
        )
        .encode(
            x=alt.X(
                "Abundance in Previous Time Span:Q",
                scale=alt.Scale(type="log", domain=[0.001, 1]),
            ),
            y=alt.Y(
                "Abundance in Current Time Span:Q",
                scale=alt.Scale(type="log", domain=[0.001, 1]),
            ),
        )
    )


def render_scatter_plot(orf_bundle: OrfDataset) -> OrfDataset:
    """
    Render a scatter plot for an OrfDataset.

    First ensures comparisons are sorted, then renders an interactive scatter plot visualizing
    the abundances of mutations between two time periods. This function modifies the OrfDataset
    in place by setting its scatter_plot attribute.

    The plot shows abundance values in log scale from 0.0001 to 1.0, with points
    representing amino acid changes. Points above the diagonal line indicate
    increasing abundance between timespans, while points below indicate decreasing
    abundance.

    Args:
        orf_bundle (OrfDataset): OrfDataset containing the data to plot

    Returns:
        OrfDataset: The input OrfDataset with its chart attribute populated
    """
    orf_bundle.render_scatter_plot()
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


def render_for_quarto(compiled_datasets: list[OrfDataset], orf_label: str) -> None:
    """
    Extract an Altair chart for a specific ORF from a list of OrfDataset objects.

    This function is used by Quarto output to display charts for specific ORFs.
    If the requested ORF does not exist, it will display an empty chart using
    ORF1 as a template.

    Args:
        compiled_datasets (list[OrfDataset]): List of OrfDataset objects containing charts
        orf_label (str): The ORF label to extract and display

    Returns:
        None: Displays the chart directly
    """
    current_bundle = [dataset for dataset in compiled_datasets if dataset.orf == orf_label]

    if len(current_bundle) != 1:
        fallback_bundle = [dataset for dataset in compiled_datasets if dataset.orf == "ORF1"]
        assert len(fallback_bundle) == 1
        unwrapped = fallback_bundle[0]

        dummy_dataset = OrfDataset(orf=unwrapped.orf, df=unwrapped.df)
        dummy_dataset.sort_comparisons()
        dummy_dataset.df = dummy_dataset.df.clone().clear()

        emptied_dataset = render_scatter_plot(dummy_dataset)
        assert emptied_dataset.chart is not None
        emptied_dataset.chart.show()
    else:
        assert current_bundle[0].chart is not None
        current_bundle[0].chart.show()


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
    orf_datasets = parse_plotting_file(all_orf_tsv, "data/major_lineages.json")

    # use the parsed dataframes wrapped in OrfDataset objects to render each plot, wrapping that in
    # OrfDataset as well
    for orf_dataset in orf_datasets:
        orf_dataset.sort_comparisons()
        orf_dataset.render_scatter_plot()
        assert orf_dataset.chart is not None

    # for each fine dataset, write out the rendered plot in the requested format
    for orf_dataset in orf_datasets:
        write_rendered_plot(orf_dataset, output_dir)


if __name__ == "__main__":
    main()
