import datetime
from datetime import date
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta
from great_tables import GT, nanoplot_options
from loguru import logger


def convert_table_to_greatable(df_path: Path) -> GT:
    """
    Converts a Polars DataFrame to an aggregated form with metadata per study_accession.

    Args:
        df (pl.DataFrame): Input DataFrame containing study_accession and other columns

    Returns:
        pl.DataFrame: Aggregated DataFrame with metadata per study
    """
    df = pl.read_csv(df_path)

    df = df.with_columns([
        pl.col("collection_date")
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)  # Convert to Date format (flexible parsing)
        .dt.strftime("%Y-%m")  # Format it as "YYYY-MM"
        .alias("year-month")
    ])

    # Group by study_accession and aggregate all columns as lists
    aggregated_df = df.group_by("study_accession").agg([
        pl.col("*").exclude("study_accession"),
    ])

    #logger.warning(aggregated_df.head())

    # Generate a month dict to use for referencing the nanoplot x value
    # The key is the month and the value is the # months since COVID
    def generate_months_dict():
        # Set the start date to January 2020
        start_date = date(2020, 1, 1)

        # Get the current date
        current_date = date.today()

        # Initialize an empty dictionary to store months
        months_dict = {}

        # Counter to track month order
        month_counter = 1

        # Iterate through months from start_date to current_date
        while start_date <= current_date:
            # Format the key as "YYYY-MM"
            key = start_date.strftime("%Y-%m")

            # Store the month with its counter value
            months_dict[key] = month_counter

            # Move to the next month
            start_date += relativedelta(months=1)
            month_counter += 1

        return months_dict

    months_dict = generate_months_dict()

    # Function to transform the month-year list into a flipped dictionary
    def month_counts(month_year):
        numbered_data = {x: 0 for x in months_dict.values()}
        for element in month_year:
            if element is None:
                continue
            # skip dates before pandemic
            if element in months_dict.keys():
                months_since_COVID = months_dict[element]
                numbered_data[months_since_COVID] = numbered_data.get(months_since_COVID, 0) + 1
            else: pass
                #print(element)
        #if len(numbered_data.values()) == 0:
        #    numbered_data[1] = 0

        return {"x": list(numbered_data.keys()), "y": list(numbered_data.values())}

    aggregated_df = aggregated_df.with_columns(
        pl.col("year-month").map_elements(month_counts).alias("month_counts")
    )

    logger.warning(aggregated_df["month_counts"])

    #logger.info(aggregated_df.columns)

    # Add total_submissions column by counting elements in any list column
    aggregated_df = aggregated_df.with_columns([
        pl.col("run_accession").list.len().alias("Total Submissions"),
        # Create SRA link column for each bioproject
        pl.col("study_accession").alias("bioproj_link").map_elements(lambda x: f"https://www.ncbi.nlm.nih.gov/bioproject/?term={x}"),
        pl.col("collection_date")
                  .list.eval(pl.element().cast(pl.Date))  # Convert strings to dates
                  .list.min()
                  .alias("Earliest Submission"),
        pl.col("collection_date")
                .list.eval(pl.element().cast(pl.Date))  # Convert strings to dates
                .list.max()
                .alias("Most Recent Submission"),
        pl.col("fastq_bytes_right")
                .list.sum().alias("Sequencing File Sizes"),
        pl.col("base_count_right")
                .list.sum().alias("Total Bases Sequenced"),
        pl.col("study_title").list.unique().list.len().alias("study_title_counts"),
    ])

    #print(aggregated_df.head())

    # Ensure each study has only one unique title
    assert (aggregated_df.select(pl.col("study_title_counts") > 1).sum().row(0)[0] == 0 and  # noqa: PT018
            aggregated_df.select(pl.col("study_title_counts") == 0).sum().row(0)[0] == 0), "Found studies with multiple or missing titles"

    # Convert study_title list to string and create markdown-style link
    aggregated_df = aggregated_df.with_columns([
        pl.col("study_title").list.first().alias("title_text"),
        # Create markdown-style link combining the title and bioproject link
        pl.struct(["study_title", "bioproj_link"]).map_elements(
            lambda x: f"[{x['study_title'][0]}]({x['bioproj_link']})",
        ).alias("Study Title"),
    ])

    # Rename for display
    aggregated_df = aggregated_df.rename({"month_counts": "Submissions Per Month Since Jan 2020"})

    # Select final columns
    aggregated_df = aggregated_df.select([
        "Study Title",
        "Total Submissions",
        #"bioproj_link",
        "Sequencing File Sizes",
        "Total Bases Sequenced",
        "Earliest Submission",
        "Most Recent Submission",
        "Submissions Per Month Since Jan 2020",
    ])

    # Sort dataframe by total_bases_sequenced in descending order
    aggregated_df = aggregated_df.sort(by="Total Bases Sequenced", descending=True)

    today = datetime.date.today()  # noqa: DTZ011
    # This step comes last
    submitters = (
        GT(aggregated_df, rowname_col="Study Title")
        .tab_header(title="Sequence Read Archive SARS-CoV-2 Wastewater Submissions", subtitle=f"Last Updated: {today}")
        .tab_spanner(label="Dates", columns=[
            "Earliest Submission",
            "Most Recent Submission"])
        .tab_spanner(label="Submission Metadata", columns=[
            "Total Submissions",
            "Sequencing File Sizes",
            "Total Bases Sequenced"])
        .tab_stubhead(label="Study Title")
        .fmt_bytes(columns=["Sequencing File Sizes"])
        .fmt_integer(columns=["Total Bases Sequenced"], compact=True)
        .fmt_markdown(columns=["Study Title"])
        .fmt_nanoplot(
            columns="Submissions Per Month Since Jan 2020",
            options=nanoplot_options(
                data_line_type= "curved",
                data_point_radius=5,
                data_point_stroke_color="black",
                data_point_stroke_width=1,
                data_point_fill_color="white",
                data_line_stroke_color="brown",
                data_line_stroke_width=2,
                data_area_fill_color="red",
                vertical_guide_stroke_color="white",
                data_bar_negative_fill_color="white")
            )
        .cols_align(align="center")
        .cols_width(
            cases={
                "Study Title": "35%",
                "Submissions Per Month Since Jan 2020": "15%"
            },
        )
    )

    return submitters

#    with open(f"acknowledgments_{today}.html", "w") as out_html:
#        out_html.write(submitters.as_raw_html(make_page=True))
