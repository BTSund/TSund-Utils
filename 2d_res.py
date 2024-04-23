#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:27:13 2024

@author: btsund
"""


import numpy as np
import matplotlib.pyplot as plt

def distance(p1, p2, box):
    d = p2 - p1
    d = d - np.round(d / box) * box
    return np.linalg.norm(d)

def radial_distribution_function(pos, res, box, dr=0.1, rcut=10):
    natoms = len(pos[0])
    rdf = np.zeros(int(rcut / dr))
    count = np.ones(int(rcut / dr))
    for i in range(len(pos)):
        for j in range(natoms):
            for k in range(j, natoms):
                if k!=j:
                    r = distance(pos[i,j], pos[i,k], box)
                    if r < rcut:
                        rdf[int(r / dr)] += res[i,j] + res[i,k]
                        count[int(r / dr)] += 2
    plt.plot(rdf)
    plt.show()
    rdf = rdf/count
    plt.plot(count)
    plt.show()
    return rdf, count


# Directory containing the files
filename = "/Users/btsund/Downloads/all.OUTCAR.100.xyzf"
filename2 = "/Users/btsund/Downloads/compare_bF_SPC_cutoff_69408.txt"
filename2 = "/Users/btsund/Downloads/compare_bF_1500K.txt"
filename = "/Users/btsund/Downloads/1500K.OUTCAR.xyzf"
filename = "/Users/btsund/Downloads/all.OUTCAR.100.xyzf"
filename2 = "/Users/btsund/Downloads/compare_bF_SPC_cutoff_69408.txt"


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
RDF,count = radial_distribution_function(pos, res, ortholat)
plt.plot(RDF)
print(np.average(res))











