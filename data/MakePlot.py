#!/bin/env python3
import os
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
import gzip


triweeks = ('W1', 'W2')
for file in os.listdir(os.getcwd()):
    if file.endswith(".plot.tsv.gz"):

        print(file)
        orf = file.split(".")[0]
        df = pd.read_csv(file, sep='\t', encoding='utf-8', header=None, names=['Change', 'Vars', 'W1', 'W2']) # pd.DataFrame(plot_dict[orf].items(), columns=['Change', 'abunds'])

        for triweek in triweeks:

            df[triweek] = pd.to_numeric(df[triweek])
            df['y'] = np.where(df[triweek] < .01, .01, df[triweek])
            df[triweek] = df['y']
            df[triweek] = pd.to_numeric(df[triweek])
        
        # plotly
        fig = px.scatter(df, x = triweeks[0], y = triweeks[1], hover_name="Change", hover_data="Vars", log_x=True, log_y=True, range_x=[.01 , 1], range_y = [.01 , 1]) # color = "Change",
        fig.add_shape(
            type="line",
            x0=.01,
            y0=.01,
            x1=1,
            y1=1,
            line_dash='dash',
        )
        fig.update_layout(
        hoverlabel_align = 'auto',
        ) 

        fig.write_html(f"{orf}.px.html")
        fig.write_image(f"{orf}.px.svg")


        # Assume df, triweeks (a list of two column names), and orf are already defined.
        # For example:
        # triweeks = ['week1', 'week2']
        # df = pd.read_csv("your_data.csv")
        # orf = "my_plot"

        # Create the scatter plot
        scatter = alt.Chart(df).mark_circle(size=60).encode(
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

        chart = chart.properties(
         width=800,
         height=700
        ).configure(autosize="fit")

        # chart = chart.autosize(
         # type='fit',
         # resize=True,
         # contains='padding'
         # )

        # Save the chart to an HTML file
        chart.save(f"{orf}.alt.html")

exit()