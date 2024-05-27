# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:31:56 2024
@author: snagchowdh
"""

# Import necessary libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Dimensions of the 2D lattice
rows = 10
cols = 10
n = rows * cols

# Values for the diagonals
g = 1.0
w = 5e-6
s = g / 2 - w

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
            beta[index, index + cols] = -w if i % 2 == 0 else -s
            beta[index + cols, index] = -w if i % 2 == 0 else -s

# Add main diagonal values
np.fill_diagonal(beta, main_diag)

# Create the graph corresponding to the adjacency matrix beta
G = nx.from_numpy_array(beta, create_using=nx.DiGraph)

# Define the positions for a 2D grid layout
pos = {(i * cols + j): (j, -i) for i in range(rows) for j in range(cols)}

# Determine the nodes to delete from the middle of the grid
#nodes_to_delete = [(4, 5), (4, 4), (5, 5), (5, 4)]

# Determine the nodes to delete from the edge of the grid
nodes_to_delete = [(0, 5), (0, 4), (1, 5), (1, 4)]

#nodes_to_delete = [(0, 5), (0, 4), (1, 5), (1, 4),(4, 5), (4, 4), (5, 5), (5, 4)]


# Remove nodes and corresponding edges from the graph
for node in nodes_to_delete:
    idx = node[0] * cols + node[1]
    G.remove_node(idx)

# Update the position dictionary to remove deleted nodes
pos = {k: v for k, v in pos.items() if k not in [node[0] * cols + node[1] for node in nodes_to_delete]}

# Remove nodes and corresponding edges from the adjacency matrix beta
for node in nodes_to_delete:
    idx = node[0] * cols + node[1]
    beta[idx, :] = 0  # Set the row corresponding to the node to zero
    beta[:, idx] = 0  # Set the column corresponding to the node to zero

# Define the system of differential equations
def system(t, y, epsilon, beta):
    x = y[:n]
    dxdt = y[n:]
    
    ddxdt = epsilon * (1 - x**2) * dxdt - beta @ x
    dydt = np.concatenate((dxdt, ddxdt))
    return dydt

# Parameters
epsilon = 0.1
duration = 2000
step_length = 0.01
t_span = (0, duration)
t_eval = np.arange(0, duration + step_length, step_length)

# Initial conditions
x0 = np.zeros(n)
x0[0] = 2.0
dx0 = np.zeros(n)
initial_conditions = np.concatenate((x0, dx0))

# Solve the system
sol = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, args=(epsilon, beta), method='RK45')

# Extract the final time step values
x_final = sol.y[:n, -1]
dx_final = sol.y[n:, -1]

# Calculate the phase angles for remaining nodes
x_final_remaining = np.delete(x_final, [node[0] * cols + node[1] for node in nodes_to_delete])
dx_final_remaining = np.delete(dx_final, [node[0] * cols + node[1] for node in nodes_to_delete])
phi_remaining = np.arctan2(dx_final_remaining, x_final_remaining)

# Update the phi array to include only the phase angles for remaining nodes
phi = phi_remaining

# Calculate the time-averaged node frequency
dphi_dt = np.gradient(np.unwrap(np.arctan2(sol.y[n:], sol.y[:n])), axis=1) / step_length
omega = np.mean(dphi_dt, axis=1)

# Find the maximum absolute frequency
omega_max = np.max(np.abs(omega))

# Normalize the frequencies by the maximum absolute frequency
omega_normalized = np.abs(omega) / omega_max

# Plot the grid with nodes colored by phase angle and background by frequency
plt.figure(figsize=(12, 12))
ax = plt.gca()

# Plot the grid background colored by normalized frequency
x_grid, y_grid = np.meshgrid(range(cols), range(rows))
freq_plot = plt.pcolormesh(x_grid, -y_grid, omega_normalized.reshape((rows, cols)), cmap='terrain_r', shading='auto', vmin=0, vmax=1)

# Plot the graph with nodes colored by phase angle with 'viridis' colormap
nodes = nx.draw_networkx_nodes(G, pos, node_color=phi, cmap='viridis', node_size=700, vmin=-np.pi, vmax=np.pi)
nx.draw_networkx_edges(G, pos, edge_color='black')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Add colorbars
cbar_freq = plt.colorbar(freq_plot, ax=ax, label=r'Normalized Frequency ($\omega / \omega_{\mathrm{max}}$)', orientation='vertical', pad=0.02)
cbar_phi = plt.colorbar(nodes, ax=ax, label='Phase Angle (radians)', orientation='vertical', pad=0.02)
cbar_phi.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar_phi.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

# Set plot labels and title
plt.title('Snapshot of VdP Network Dynamics\nNodes Colored by Phase Angle (ϕ)\nBackground Colored by Normalized Time-Averaged Node Frequency ($\omega / \omega_{\mathrm{max}}$)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.gca().invert_yaxis()  # Invert y-axis to match matrix layout
plt.show()
