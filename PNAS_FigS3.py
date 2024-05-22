# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:46:15 2024

@author: snagchowdh
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker

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
            # T2 configuration
            beta[index, index + cols] = -w if i % 2 == 0 else -s
            beta[index + cols, index] = -w if i % 2 == 0 else -s
            
            # # T1 configuration
            # beta[index, index + cols] = -s if i % 2 == 0 else -w
            # beta[index + cols, index] = -s if i % 2 == 0 else -w

# Add main diagonal values
np.fill_diagonal(beta, main_diag)

# Create the graph corresponding to the adjacency matrix beta
G = nx.from_numpy_array(beta, create_using=nx.DiGraph)

# Define the positions for a 2D grid layout
pos = {(i * cols + j): (j, -i) for i in range(rows) for j in range(cols)}

plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', edge_color='black')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Graph corresponding to the adjacency matrix beta (2D Lattice)')
plt.show()

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

# Calculate the phase angles
phi = np.arctan2(dx_final, x_final)

# Calculate the time-averaged node frequency
dphi_dt = np.gradient(np.unwrap(np.arctan2(sol.y[n:], sol.y[:n])), axis=1) / step_length
omega = np.mean(dphi_dt, axis=1)

# Find the index of the maximum absolute frequency
max_idx = np.argmax(np.abs(omega))

# Store the maximum absolute frequency along with its sign
omega_max = omega[max_idx] 

# Normalize the frequencies by the maximum frequency
omega_normalized = omega / omega_max

# Plot the grid with nodes colored by phase angle and background by frequency
plt.figure(figsize=(12, 12))
ax = plt.gca()

# Plot the grid background colored by normalized frequency
x_grid, y_grid = np.meshgrid(range(cols), range(rows))

# Plot the grid background colored by normalized frequency with 'turbo_r' colormap
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
