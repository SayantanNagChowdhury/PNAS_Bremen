# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:45:04 2024

@author: snagchowdh
"""

import numpy as np

# Order of the matrix
n = 6

# Values for the diagonals
g = 1.0   # example value for diagonal entries
w = 0.003   # example value for upper and lower diagonal entries
s = g-w   # example value for upper and lower diagonal entries

# Create the main diagonal
main_diag = np.full(n, g)

# Create the alternating upper and lower diagonals
upper_diag = np.array([-w if i % 2 == 0 else -s for i in range(n-1)])
lower_diag = np.array([-w if i % 2 == 0 else -s for i in range(n-1)])

# Construct the tridiagonal matrix
matrix = np.diag(main_diag) + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)

print(matrix)
