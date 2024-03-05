#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:50:42 2024

@author:  btsund
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

n = 0
data = pd.read_csv('C:/Users/16207/Downloads/chimes_scan_3b.type_0.dat',
                   delimiter='\s+')  # change to the path to the file
df = data.to_numpy()
dr = float(data.columns[3])  # sets the step size in ang
outer_cutoff = float(data.columns[2]) * 10  # outer cutoff in pm
inner_cutoff = float(data.columns[1]) * 10  # inner cutoff in pm
math = df.shape[0]  # total length of the column
iterations = int(round((math * dr ** 3) ** (1 / 3) / dr, 1))
volume = np.ones([iterations, iterations, iterations])
sort = data.sort_values(by=[str(dr)], ascending=True)
minval = df[sort.index[int(0)], 3]
maxval = df[sort.index[-1], 3]
for x in range(iterations):
    for y in range(iterations):
        for z in range(iterations):
            volume[x, y, z] = df[n, 3]  # changes the shape to be "volumetric"
            n += 1
nb_frames, r, c = volume.shape # all of these are the same, so I could theoretically change all of them to be iterations

# Define frames

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=volume[:,:,k],
    surfacecolor=(volume[k]),
    y=np.linspace(inner_cutoff, outer_cutoff, num=iterations + 1),
    x=np.linspace(inner_cutoff, outer_cutoff, num=iterations + 1),
    cmin=-maxval, cmax=maxval),
    name=str(k))
    for k in range(int(outer_cutoff - inner_cutoff))])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=volume[:,:,0],
    y=np.linspace(inner_cutoff, outer_cutoff, num=iterations + 1),
    x=np.linspace(inner_cutoff, outer_cutoff, num=iterations + 1),
    surfacecolor=(volume[0]),
    cmin=-maxval, cmax=maxval, colorscale='RdBu_r' ,
    colorbar=dict(thickness=20, ticklen=4)))


def frame_args(duration):
    return {"frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"}, }


sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]

# Layout
fig.update_layout(
    title='3 Body Energy visualization (kcal/mol/atom)',
    width=900,
    height=900,
    scene=dict(
        zaxis=dict(range=[-min(abs(minval), abs(maxval)), min(abs(minval), abs(maxval))], autorange=False),
        yaxis=dict(nticks=4, range=[inner_cutoff, outer_cutoff], ),
        xaxis=dict(nticks=4, range=[inner_cutoff, outer_cutoff], ), ),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders
)

fig.update_layout(scene=dict(
    xaxis_title='1, 2 Distance (pm)',
    yaxis_title='1, 3 Distance (pm)',
    zaxis_title='2, 3 Distance (pm)',
    aspectmode='cube'),
    yaxis=dict(nticks=4, range=[inner_cutoff, outer_cutoff], ),
    xaxis=dict(nticks=4, range=[inner_cutoff, outer_cutoff], ),

)
fig.show()
