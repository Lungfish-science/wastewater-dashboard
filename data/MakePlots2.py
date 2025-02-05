#!/bin/env python3
import os
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
import gzip


file = "VariantPMs.tsv"
df = pd.read_csv(file, sep='\t', encoding='utf-8') # , header=None, names=['Change', 'Vars', 'W1', 'W2'])
df = df.iloc[1:] 

df = df.iloc[: , [2, 3, 4, 6, 9]].copy()
df.columns.values[0] = "ORF"
df.columns.values[1] = "Change"
df.columns.values[2] = "Vars"

triweeks = df.columns.values[3:]

for triweek in triweeks:

    df[triweek] = pd.to_numeric(df[triweek])
    df['y'] = np.where(df[triweek] < .01, .01, df[triweek])
    df[triweek] = df['y']
    df[triweek] = pd.to_numeric(df[triweek])

df.drop('y', axis=1, inplace=True)


orfs = set(df["ORF"].tolist())

for orf in orfs:

    # Create the scatter plot
    scatter = alt.Chart(df[df['ORF'] == orf]).mark_circle(size=60).encode(
     x=alt.X(f'{triweeks[0]}:Q',
     scale=alt.Scale(type='log', domain=[0.01, 1]),
     title=triweeks[0]),
     y=alt.Y(f'{triweeks[1]}:Q',
     scale=alt.Scale(type='log', domain=[0.01, 1]),
     title=triweeks[1]),
     color=alt.Color('Change:N'),
     tooltip=['Change', 'Vars']
    )

    # Create data for the dashed line from (0.01, 0.01) to (1, 1)
    line_data = pd.DataFrame({
     triweeks[0]: [0.01, 1],
     triweeks[1]: [0.01, 1]
    })

    # Create the dashed line
    line = alt.Chart(line_data).mark_line(
     strokeDash=[5, 5], # This sets a dashed style
     color='black'
    ).encode(
     x=alt.X(f'{triweeks[0]}:Q',
     scale=alt.Scale(type='log', domain=[0.01, 1])),
     y=alt.Y(f'{triweeks[1]}:Q',
     scale=alt.Scale(type='log', domain=[0.01, 1]))
    )

    # Combine the scatter plot and the line into one chart
    chart = scatter + line

    chart = chart.interactive().properties(width="container", height=500)

    # Save the chart to an HTML file
    chart.save(f"{orf}.html")

exit()