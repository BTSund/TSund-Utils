#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:45:35 2024

@author: btsund
"""


import numpy as np
import sys
import plotly.graph_objects as go
from sys import exit
import matplotlib.pyplot as plt

listx = [-1]
listy = [-1]
masterx = []
mastery = []
y2 = []
n = 0
i = 0
IFSTREAM = open("/Users/btsund/Downloads/300AAAA_3b_lowfreq.xyzf", 'r')
IFSTREAM2 = open("/Users/btsund/Downloads/compare.txt", 'r')
listr23 = [30.0]
yres = 0
xres = 0
EXIT_CONDITION = False
countx = []
county = []
reslist = []
res2 = []
res3 = []
while not EXIT_CONDITION:
    LINE = IFSTREAM.readline()
    if not LINE:
        print("  End of file reached ")
        break
    if i == 0:
        natoms = int(LINE)
        pos = np.zeros([3, natoms])
        i += 1
        continue
    if i == 1:
        i += 1
        n = 0
        continue
    if i == 2:
        for j in range(natoms):
            pos[j, 0] = round(float(LINE.split()[1]),3)
            pos[j, 1] = round(float(LINE.split()[2]),3)
            pos[j, 2] = round(float(LINE.split()[3]),3)
            #print(pos[:, j])
            if j != natoms-1:
                LINE = IFSTREAM.readline()
        LINE2 = IFSTREAM2.readline()
        LINE2 = IFSTREAM2.readline()
        LINE2 = IFSTREAM2.readline()
        residual = abs(float(LINE2.split()[0])-float(LINE2.split()[1]))
        if round(pos[2, 1],2) > listy[-1]:
            listy.append(round(pos[2,1],2))
            reslist.append(residual)
        if round(pos[2, 0], 2) > listx[-1]:
          #  print(pos[2, 0])
            mastery.append(listy)
            listy = []
            res2.append(reslist)
            reslist = []
            listx.append(pos[2,0])
            listy.append(round(pos[2,1],2))
            reslist.append(residual)
        if pos[1, 0] not in listr23:
            mastery.append(listy)
            y2.append(mastery)
            mastery = []
            listy = []
            listy.append(pos[2,1])
            res2.append(reslist)
            res3.append(res2)
            res2=[]
            masterx.append(listx)
            listx = []
            listx.append(pos[2,0])
            listr23.append(pos[1, 0])
        i = 0
countx.append(len(listx))
masterx.append(listx)
county.append(len(listy))
mastery.append(listy)
y2.append(mastery)
res2.append(reslist)
res3.append(res2)
IFSTREAM.close()
county = county[1:]
mastery = mastery[1:]
masterx[0] = masterx[0][1:]
y2[0][:] = y2[0][1:]
res3[0][:] = res3[0][1:]
max_lengths = [max(len(sublist) for sublist in inner_list) for inner_list in y2]
result_list = []
for inner_list, max_length in zip(y2, max_lengths):
    # Create an empty numpy array filled with NaN
    result_array = np.full((len(inner_list), max_length), np.nan)

    # Fill in values from the sublists into the result array
    for i, sublist in enumerate(inner_list):
        result_array[i, :len(sublist)] = sublist

    result_list.append(result_array)

max_length = max(len(sublist) for sublist in masterx)
result_arrayx = np.full((len(masterx), max_length), np.nan)
for i, sublist in enumerate(masterx):
    result_arrayx[i, :len(sublist)] = sublist
xlist = []
for i in range(len(result_list)):
    xlist.append(result_arrayx)
xs = []
for i in range(len(listy)):
    xs.append(np.array(listx))
test = np.array(xs).transpose()
x_final = []
for i in range(len(listr23)):
    x_final.append(test)
max_lengths = [max(len(sublist) for sublist in inner_list) for inner_list in res3]
result_listres = []
for inner_list, max_length in zip(res3, max_lengths):
    # Create an empty numpy array filled with NaN
    result_array = np.full((len(inner_list), max_length), np.nan)

    # Fill in values from the sublists into the result array
    for i, sublist in enumerate(inner_list):
        result_array[i, :len(sublist)] = sublist

    result_listres.append(result_array)
print(x_final[0][-1, -1])
print(result_list[0][-1, -1])
print(result_listres[0][-1, -1])
fig = go.Figure(frames=[go.Frame(data=[go.Surface(
    z=result_listres[k],
    x=x_final[k],
    y=result_list[k]),
    go.Scatter3d(x=[-k/2, k/2], y=[0, 0], z=[0, 0], mode='markers')],
    name=str(k))
    for k in range(11)])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=result_listres[0],
    x=x_final[0],
    y=result_list[0]))


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
        camera=dict(projection=dict(type='orthographic')),
        #camera=dict(projection=dict(type='perspective')),
        zaxis=dict(range=[0, 100], autorange=True),
        yaxis=dict(nticks=4, range=[0, 100], autorange=False),
        xaxis=dict(nticks=4, range=[0, 85], autorange=False),
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
