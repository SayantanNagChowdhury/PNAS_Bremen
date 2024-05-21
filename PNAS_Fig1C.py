# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:56:08 2024

@author: snagchowdh
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Dimensions of the 2D lattice
rows = 4
cols = 4
n = rows * cols

# Values for the diagonals
g = 1.0   # example value for diagonal entries
w = 0.003   # example value for upper and lower diagonal entries
s = g - w   # example value for upper and lower diagonal entries

# Create the main diagonal
main_diag = np.full(n, g)

# Initialize the adjacency matrix
beta = np.zeros((n, n))

# Fill in the adjacency matrix for a 2D lattice
for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        if j < cols - 1:  # Right neighbor
            beta[index, index + 1] = -w if j % 2 == 0 else -s
            beta[index + 1, index] = -w if j % 2 == 0 else -s
        if i < rows - 1:  # Bottom neighbor
        
            # T2 configuration
            beta[index, index + cols] = -w if i % 2 == 0 else -s
            beta[index + cols, index] = -w if i % 2 == 0 else -s
            
            # # T1 configuration
            # beta[index, index + cols] = -s if i % 2 == 0 else -w
            # beta[index + cols, index] = -s if i % 2 == 0 else -w

# Add main diagonal values
np.fill_diagonal(beta, main_diag)

# print("Adjacency matrix beta for the 2D lattice:")
# print(beta)

# Create the graph corresponding to the adjacency matrix beta
G = nx.from_numpy_array(beta, create_using=nx.DiGraph)

# Define the positions for a 2D grid layout
pos = {(i * cols + j): (j, -i) for i in range(rows) for j in range(cols)}

plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', edge_color='black')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Graph corresponding to the adjacency matrix beta (2D Lattice)')
plt.show()
