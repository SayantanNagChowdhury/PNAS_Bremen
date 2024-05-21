# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:07:49 2024

@author: snagchowdh
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx

# Define the tridiagonal matrix
n = 6
g = 1.0
w = 0.003
s = g - w

# Construct the tridiagonal matrix
main_diag = np.full(n, g)

#Construction of 1D chain with weak-strong-weak
upper_diag = np.array([-w if i % 2 == 0 else -s for i in range(n-1)])
lower_diag = np.array([-w if i % 2 == 0 else -s for i in range(n-1)])

# #Construction of 1D chain with strong-weak-strong
# upper_diag = np.array([-s if i % 2 == 0 else -w for i in range(n-1)])
# lower_diag = np.array([-s if i % 2 == 0 else -w for i in range(n-1)])


beta = np.diag(main_diag) + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)

# Print the matrix beta
print("Matrix beta:")
print(beta)

# Define the system of differential equations
def system(t, y, epsilon, beta):
    x = y[:n]
    dxdt = y[n:]
    
    ddxdt = epsilon * (1 - x**2) * dxdt - beta @ x
    dydt = np.concatenate((dxdt, ddxdt))
    return dydt

# Parameters
epsilon = 0.1 #0.0
duration = 2000
step_length = 0.01
t_span = (0, duration)
t_eval = np.arange(0, duration + step_length, step_length)

#Initial conditions
x0 = np.zeros(n)
x0[0] = 2.0
dx0 = np.zeros(n)
initial_conditions = np.concatenate((x0, dx0))


# # Define the range for initial conditions
# x_min, x_max = -4.0, 4.0
# dx_min, dx_max = -0.4, 0.4

# # Generate random initial conditions for x_i(0) and \dot{x}_i(0)
# x0 = np.random.uniform(x_min, x_max, n)
# dx0 = np.random.uniform(dx_min, dx_max, n)

# # Concatenate x0 and dx0 to form the new initial conditions
# initial_conditions = np.concatenate((x0, dx0))


# Solve the system
sol = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, args=(epsilon, beta), method='RK45')

# Extract the solution for the last 500 time units
time_interval = 500 #500
time_indices = (sol.t >= duration - time_interval)

# Get the solution for x_i (the first n elements of y) over the last 500 time units
x_last_500 = sol.y[:n, time_indices]
time_last_500 = sol.t[time_indices]

# Plot the variation of x_i over the last 500 time units
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(time_last_500, x_last_500[i], label=f'x_{i+1}')
plt.xlabel('Time')
plt.ylabel('x_i')
plt.title('Variation of x_i over the last 500 time units')
plt.legend()
plt.grid(True)
plt.show()

# Plot each x_i in a separate subfigure within a single figure
plt.figure(figsize=(12, 8))
for i in range(n):
    plt.subplot(n, 1, i+1)
    plt.plot(time_last_500, x_last_500[i])
    plt.xlabel('Time')
    plt.ylabel(f'x_{i+1}')
    plt.title(f'Variation of x_{i+1} over the last 500 time units')
    plt.grid(True)

plt.tight_layout()  # Adjust subplot layout to make it more readable
plt.show()


# # Plot the graph corresponding to the adjacency matrix beta
# G = nx.from_numpy_array(beta, create_using=nx.DiGraph)

# plt.figure(figsize=(8, 8))
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', edge_color='black')
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.title('Graph corresponding to the adjacency matrix beta')
# plt.show()

# Plot the graph corresponding to the adjacency matrix beta in a chain layout
G = nx.from_numpy_array(beta, create_using=nx.DiGraph)

plt.figure(figsize=(8, 3))
pos = {i: (i, 0) for i in range(n)}  # Specify the positions of nodes in a chain layout
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', edge_color='black')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Graph corresponding to the adjacency matrix beta (Chain Layout)')
plt.show()

# Extract the solution for the first 500 time units
time_interval = 500 #2000
time_indices = (sol.t <= time_interval)

# Get the solution for x_i (the first n elements of y) over the first 500 time units
x_first_500 = sol.y[:n, time_indices]
dx_first_500 = sol.y[n:, time_indices]
time_first_500 = sol.t[time_indices]

# Plot x_5 vs dx_5 and x_3 vs dx_3 with time t as color in subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot x_5 vs dx_5 with time t as color
sc1 = ax1.scatter(x_first_500[5], dx_first_500[5], c=time_first_500, cmap='viridis')
ax1.plot(x_first_500[5][0], dx_first_500[5][0], 'o', label='Initial condition', color='red')
ax1.set_xlabel('x_5')
ax1.set_ylabel('dx_5')
ax1.set_title('x_5 vs dx_5 with Time as Color (First 500 Time Units)')
ax1.legend()
ax1.grid(True)
fig.colorbar(sc1, ax=ax1, label='Time')

# Plot x_3 vs dx_3 with time t as color
sc2 = ax2.scatter(x_first_500[3], dx_first_500[3], c=time_first_500, cmap='viridis')
ax2.plot(x_first_500[3][0], dx_first_500[3][0], 'o', label='Initial condition', color='red')
ax2.set_xlabel('x_3')
ax2.set_ylabel('dx_3')
ax2.set_title('x_3 vs dx_3 with Time as Color (First 500 Time Units)')
ax2.legend()
ax2.grid(True)
fig.colorbar(sc2, ax=ax2, label='Time')

plt.tight_layout()
plt.show()


