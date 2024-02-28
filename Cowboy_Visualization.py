#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:50:42 2024

@author:  btsund
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def find(df, c, dr, outcut, incut):
    c = round(c, 4)
    energy = np.ones([int(np.ceil((2 * outcut) / dr)), int(np.ceil((2 * outcut) / dr))])
    print(energy.shape)
    for x in np.arange(-outcut, outcut, dr):
        x = round(x, 4)
        for y in np.arange(-outcut, outcut, dr):
            y = round(y, 4)
            a = (x ** 2 + y ** 2) ** (.5)
            b = (x ** 2 + (c - y) ** 2) ** (.5)
            if (a >= outcut) or (b >= outcut):
                energy[int(round((x + outcut) / dr)), int(round((y + outcut) / dr))] = float('NaN')
                continue
            if (a <= incut) or (b <= incut):
                energy[int(round((x + outcut) / dr)), int(round((y + outcut) / dr))] = float('NaN')

                continue
            # print(int(np.floor((a-incut)/dr)-1))
            if int(np.ceil((a - incut) / dr)) == df.shape[0] or int(np.ceil((b - incut) / dr)) == df.shape[0]:
                energy[int(round((x + outcut) / dr)), int(round((y + outcut) / dr))] = float('NaN')
                continue
            energy[int(round((x + outcut) / dr)), int(round((y + outcut) / dr))] = \
                df[int(np.floor((a - incut) / dr)), int(np.floor((b - incut) / dr)), int((c - incut) / dr)] + \
                ((a - incut) / dr - np.floor((a - incut) / dr)) * \
                (df[int(np.ceil((a - incut) / dr)), int(np.floor((b - incut) / dr)), int((c - incut) / dr)] -
                 df[int(np.floor((a - incut) / dr)), int(np.floor((b - incut) / dr)), int((c - incut) / dr)]) + \
                ((b - incut) / dr - np.floor((b - incut) / dr)) * \
                (df[int(np.floor((a - incut) / dr)), int(np.ceil((b - incut) / dr)), int((c - incut) / dr)] -
                 df[int(np.floor((a - incut) / dr)), int(np.floor((b - incut) / dr)), int((c - incut) / dr)])
                  # would usually be over (x2-x1) but is = 1 always

    return energy


n = 0
data = pd.read_csv('/Users/btsund/Downloads/chimes_scan_3b.type_0 (2).dat', sep='\s+')
df = data.to_numpy()
dr = float(data.columns[3])
outer_cutoff = float(data.columns[2])
inner_cutoff = float(data.columns[1])
math = df.shape[0]
iterations = int(round((math * dr ** 3) ** (1 / 3) / dr, 4))
volume = np.ones([iterations, iterations, iterations])
sort = data.sort_values(by=[data.columns[3]], ascending=True)
minval = df[sort.index[int(math * .0025)], 3]
maxval = df[sort.index[-2], 3]
for x in range(iterations):
    for y in range(iterations):
        for z in range(iterations):
            volume[x, y, z] = df[n, 3]
            n += 1

if round(volume[0,0,1],4) != round(volume[0,1,0],4):
    if round(volume[1,0,0],4) == round(volume[0,1,0],4):
        volume = np.swapaxes(volume, 0, 2)
    else:
        volume = np.swapaxes(volume, 0, 1)
# Define frames
fig = go.Figure(frames=[go.Frame(data=[go.Surface(
    z=find(volume, round(k, 4), dr, outer_cutoff, inner_cutoff),
    x=(np.arange(-outer_cutoff, outer_cutoff, dr)-k/2),
    y=np.arange(-outer_cutoff, outer_cutoff, dr),
    cmin=-np.min([abs(maxval), abs(minval)]), cmax=np.min([abs(maxval), abs(minval)])),
    go.Scatter3d(x=[-k/2, k/2], y=[0, 0], z=[0, 0], mode='markers')],
    name=str(k))
    for k in np.arange(inner_cutoff, outer_cutoff, dr)])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=find(volume, inner_cutoff, dr, outer_cutoff, inner_cutoff),
    x=np.arange(-outer_cutoff, outer_cutoff, dr)-inner_cutoff/2,
    y=np.arange(-outer_cutoff, inner_cutoff + outer_cutoff, dr),
    cmin=-np.min([abs(maxval), abs(minval)]), cmax=np.min([abs(maxval), abs(minval)]), colorscale='RdBu_r',
    colorbar=dict(thickness=20, ticklen=4)))

fig.add_scatter3d(x=[0,inner_cutoff], y=[0, 0], z=[0, 0], mode='markers')

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
        #camera=dict(projection=dict(type='orthographic')),
        camera=dict(projection=dict(type='perspective')),
        zaxis=dict(range=[minval, maxval], autorange=False),
        yaxis=dict(nticks=4, range=[-outer_cutoff, outer_cutoff], autorange=False),
        xaxis=dict(nticks=4, range=[-outer_cutoff, outer_cutoff], autorange=False),
        aspectmode='cube',
        aspectratio=dict(x=1, y=1, z=100)),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(100)],
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
    xaxis_title='x position of atom 3',
    yaxis_title='y position of atom 3',
    zaxis_title='energy kcal/mol/atom',
    aspectratio=dict(z=10))
)
fig.show()
