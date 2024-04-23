#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:50:42 2024

@author:  btsund
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed


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
            Q11 = df[int(np.floor((a - incut) / dr)), int(np.floor((b - incut) / dr)), int((c - incut) / dr)]
            Q21 = df[int(np.ceil((a - incut) / dr)), int(np.floor((b - incut) / dr)), int((c - incut) / dr)]
            Q12 = df[int(np.floor((a - incut) / dr)), int(np.ceil((b - incut) / dr)), int((c - incut) / dr)]
            Q22 = df[int(np.ceil((a - incut) / dr)), int(np.ceil((b - incut) / dr)), int((c - incut) / dr)]
            energy[int(round((x + outcut) / dr)), int(round((y + outcut) / dr))] = \
                Q11 * (np.ceil((a - incut) / dr) - (a - incut) / dr) * (np.ceil((b - incut) / dr) - (b - incut) / dr) + \
                Q21 * ((a - incut) / dr - np.floor((a - incut) / dr)) * (np.ceil((b - incut) / dr) - (b - incut) / dr) + \
                Q12 * (np.ceil((a - incut) / dr) - (a - incut) / dr) * ((b - incut) / dr - np.floor((b - incut) / dr)) + \
                Q22 * ((a - incut) / dr - np.floor((a - incut) / dr)) * ((b - incut) / dr - np.floor((b - incut) / dr))
    return energy


def minimum_image_distance(r_i, r_j, box):
    # Compute the distance vector between two particles, applying PBC
    distance_vector = r_j - r_i
    distance_vector -= box * np.round(distance_vector / box)  # Apply minimum image convention
    distance = np.linalg.norm(distance_vector)
    return distance


def compute_distance_matrix(pos, box):
    natoms = len(pos)
    distance_matrix = np.zeros((natoms, natoms))

    for i in range(natoms):
        for j in range(i + 1, natoms):
            dist = minimum_image_distance(pos[i], pos[j], box)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix

    return distance_matrix


def compute_rdf_for_configuration(pos, box, rcut, dr, res_i):
    natoms = len(pos)
    distance_matrix = compute_distance_matrix(pos, box)

    rdf_local = np.zeros([int(rcut / dr)]*3)
    count_local = np.zeros([int(rcut / dr)]*3)

    for j in range(natoms):
        for k in range(j + 1, natoms):
            r_jk = distance_matrix[j, k]
            if r_jk > rcut:
                continue

            for h in range(k + 1, natoms):
                r_jh = distance_matrix[j, h]
                r_kh = distance_matrix[k, h]

                if r_jh > rcut or r_kh > rcut:
                    continue

                # Triangle inequalities - speeds up calculation if one side is longer than the sum of the other two

                b, c, a = sorted([r_jk, r_jh, r_kh])
                res = (res_i[j] + res_i[k] + res_i[h])
                rdf_local[int(a / dr), int(b / dr), int(c / dr)] += res
                count_local[int(a / dr), int(b / dr), int(c / dr)] += 3
                rdf_local[int(a / dr), int(c / dr), int(b / dr)] += res
                count_local[int(a / dr), int(c / dr), int(b / dr)] += 3
                rdf_local[int(b / dr), int(a / dr), int(c / dr)] += res
                count_local[int(b / dr), int(a / dr), int(c / dr)] += 3
                rdf_local[int(b / dr), int(c / dr), int(a / dr)] += res
                count_local[int(b / dr), int(c / dr), int(a / dr)] += 3
                rdf_local[int(c / dr), int(b / dr), int(a / dr)] += res
                count_local[int(c / dr), int(b / dr), int(a / dr)] += 3
                rdf_local[int(c / dr), int(a / dr), int(b / dr)] += res
                count_local[int(c / dr), int(a / dr), int(b / dr)] += 3

    return rdf_local, count_local


def main(pos, box, rcut, dr, res):
    results = Parallel(n_jobs=-1)(delayed(compute_rdf_for_configuration)(pos[i], box, rcut, dr, res[i])
                                   for i in range(len(pos)))

    rdf = np.sum([result[0] for result in results], axis=0)
    count = np.sum([result[1] for result in results], axis=0)

    return rdf, count


def load_trajectory(filepath, num_timesteps, dump_freq, num_atoms, filepath2):
    num_frames = num_timesteps // dump_freq + 1 \
        if num_timesteps % dump_freq == 0 else num_timesteps // dump_freq
    with open(filepath, 'r') as file:
        with open(filepath2, 'r') as resfile:
            for t in range(num_frames):
                try:
                   if int(t) ==0: # fix the ranges to fit your file format
                       for _ in range(1):  # Skipping metadata lines
                           num_atoms = int(next(file).split()[0])
                           positions = np.zeros((num_frames, num_atoms, 3))
                           res = np.zeros((num_frames, num_atoms))
                           cell = next(file).split()[1:11]
                   if int(t) !=0:
                       for _ in range(1):  # Skipping metadata lines
                           next(file)
                       for _ in range(1):  # Skipping metadata lines
                           cell = next(file).split()[1:11]
                   residual = 0
                   for i in range(num_atoms):
                       atom_data = next(file).split()
                       for j in range(3):
                           LINE = next(resfile).split()
                           residual += (float(LINE[1])-float(LINE[2]))**2
                       res[t,int(i)] = np.sqrt(residual)
                       residual=0
                       x, y, z = float(atom_data[1]), float(atom_data[2]), float(atom_data[3])
                       positions[t, int(i)] = [x, y, z]
                except StopIteration:
                    num_frames = t * dump_freq
                    break
    return positions, num_frames, num_atoms, cell, res

# Directory containing the files
filename = "/Users/btsund/Downloads/all.OUTCAR.100.xyzf"
filename2 = "/Users/btsund/Downloads/compare_bF_SPC_cutoff_69408.txt"


inner_cutoff = 0
outer_cutoff = 7
dr = .1
iofreq = 1
timesteps = 23 # total timesteps
pos, timesteps, atoms, cellv, res = load_trajectory(filename, timesteps, iofreq,1, filename2)
cell = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.0]])
for i in range(3):
    for j in range(3):
        cell[i,j] = float(cellv[i*3+j])
ortholat = np.array([cell[0,0],cell[1,1],cell[2,2]])
for i in range(len(pos)):
    pos[i] = pos[i] % ortholat
    pos[i] -= np.floor(pos[i] / ortholat) * ortholat
rdf, count = main(pos, ortholat, outer_cutoff, dr, res)
volume = rdf
for j in range(int(outer_cutoff / dr)):
        for k in range(int(outer_cutoff / dr)):
            for h in range(int(outer_cutoff / dr)):
                volume[j,k,h] = rdf[j,k,h] / (count[j,k,h]+1)

minval = 0
maxval = np.amax(volume, axis=None)
# Define frames
fig = go.Figure(frames=[go.Frame(data=[go.Surface(
    z=find(volume, round(k, 4), dr, outer_cutoff, inner_cutoff),
    x=(np.arange(-outer_cutoff, outer_cutoff, dr)-k/2),
    y=np.arange(-outer_cutoff, outer_cutoff, dr),
    cmin=0, cmax=maxval),
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
