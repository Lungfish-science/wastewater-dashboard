import altair as alt
import pandas as pd
from vega_datasets import data as vg_data

# ------------------------------
# Load TopoJSON state shapes
# ------------------------------
states = alt.topo_feature(vg_data.us_10m.url, 'states')
week = "2024-11-24"

# ------------------------------
# FIPS → HHS Region Mapping
# ------------------------------
hhs_data = pd.DataFrame({
    'id': [
        1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56
    ],
    'region': [
        4, 10, 9, 6, 9, 8, 1, 3, 3, 4, 4, 9, 10, 5, 5, 7, 7, 4, 6, 1,
        3, 1, 5, 5, 4, 7, 8, 7, 9, 1, 2, 6, 2, 3, 8, 5, 6, 10, 3, 1,
        4, 8, 4, 6, 8, 1, 3, 10, 3, 5, 8
    ]
})
hhs_data["region"] = hhs_data["region"].astype(str)
hhs_data_chart = alt.InlineData(values=hhs_data.to_dict(orient='records'))

# ------------------------------
# Region Selection Parameter
# ------------------------------
region_select = alt.selection_point(
    fields=['region'],
    name='region_select',
    clear='dblclick',
    empty=True # Crucial element to make sure that it default loads everything so then click can happen
)

# ------------------------------
# U.S. Map Chart
# ------------------------------

us_title = alt.TitleParams(
    "SARS-CoV-2 Variants by HHS region",
    subtitle=[f'Week of: {week}'],
)

us_chart = alt.Chart(states).mark_geoshape(
    strokeWidth=1,
    stroke='white'
).encode(
    color=alt.condition(
        region_select,
        alt.Color('region:N', legend=alt.Legend(title="HHS Region")),
        alt.value('lightgray')  # non-selected states
    ),
    opacity=alt.condition(
        region_select,
        alt.value(1.0),     # selected region = full opacity
        alt.value(0.4)      # other regions = faded
    ),
    tooltip=['region:N']
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(hhs_data_chart, key='id', fields=['region'])
).transform_filter(
    alt.FieldOneOfPredicate(field='id', oneOf=hhs_data['id'].tolist())
# ).transform_filter(
#     "datum.region == region_select"
).transform_calculate(
    region="datum.region",
).add_params(
    region_select,
).project(
    type='albersUsa'
).properties(
    width=800,
    height=500,
    title=us_title
)


# ------------------------------
# Variant Data for a Given Week
# ------------------------------
hhs_var = pd.read_csv("data/HHS.long.tsv", sep="\t")
hhs_var.rename(columns={'Region': 'region'}, inplace=True)
hhs_var["region"] = hhs_var["region"].astype(str)

first_week_df = hhs_var[hhs_var['Week'] == week]

# ------------------------------
# Variant Bar Chart (Filtered)
# ------------------------------
vars_in_week = alt.Chart(first_week_df).mark_bar().encode(
    x=alt.X("Abundance:Q", title="Abundance"),
    y=alt.Y("region:O", title="Region"),
    color=alt.Color("AA Change:N", legend=None),
    tooltip=['AA Change:N']
).transform_filter(
    region_select
).add_params(
    region_select
).properties(
    width=800,
    height=400,
    title='Variant Abundances in Selected Region'
)

# Combine the charts vertically and display
combined_chart = alt.vconcat(us_chart, vars_in_week).resolve_scale(
    color='independent'
)

combined_chart

# us_chart.show()
# vars_in_week.show()
# ------------------------------
# Combine and Show
# ------------------------------
#(us_chart & vars_in_week).show()
