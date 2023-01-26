"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from math import ceil
from copy import deepcopy

## ---------- ## DEFAULTS ## ---------- ##

'''
# Labels for plotting and index of arrays
labels = ['Water', 'Oil', 'Air']

# Color value used for color map
colors = np.array([0.2, 0.65, 0.45])

# Amount of each phase
A = np.array([0.4, 0.6, 0.5])

# Starting position for Gaussian peaks
B = np.array([0.0, 0.0, 0.0])

# Variance of Gaussian peaks
C = 2.0

# Density of each phase
D = np.array([1.0, 0.93, 1.225e-3])

# Amount of solute in each phase
# S[-1] = target solute
S = np.array([[0.95, 0.05], [0.05, 0.95]])

# Polarity of solvent
Lpol = np.array([2 * 1.24 * np.cos((109.5 / 2) * (np.pi / 180.0)), 0.0])

# Polarity of solutes
Spol = np.array([0.9, 0.1])

# Initial time variable such that Gaussians have normalized area
t0 = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

# Time step
dt = 0.05
'''

# Maximum varience of Gaussian peaks
Cmix = 2.0

# Array for x/height positions
x = np.linspace(-10.0, 10.0, 1000, endpoint=True, dtype=np.float32)

# Function to map the separation Gaussians to discretized state
def map_to_state(A, B, C, colors, x=x):
    # Create a copy of B for temporary changes
    B1 = np.copy(B)
    
    # Array for layers at each time step
    L = np.zeros(100, dtype=np.float32) + colors[-1]

    # Initialize time variable such that Gaussians have normalized area
    C = np.max([C, 1e-10])
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Number of pixels available for each phase
    sum_A = 1.0 if np.sum(A) == 0 else np.sum(A)
    n = ((A / sum_A) * L.shape[0])
    count = 0
    for i in np.where((n > 1e-8) | (n < 0.5 - 1e-8))[0]:
        n[i] = ceil(n[i])
        count += 1

    n = n.astype(int)
    if np.sum(n[:-1]) > L.shape[0]:
        for i in range(count):
            n[n.argmax()] -= 1
    else:
        # Take rounding errors into account
        n[-1] = 0
        n[-1] = L.shape[0] - np.sum(n)
    
    # Loop over each layer pixel
    for l in range(L.shape[0]):
        # Map layer pixel position to x position
        k = int(((l + 0.5) / L.shape[0]) * x.shape[0])

        # Calculate Gaussian values at current x position
        P = A * np.exp(-1.0 * (((x[k] - B) / (2.0 * C)) ** 2) + t)

        # Check to see if any phases have no pixels remaining
        past_due = False
        for j in range(P.shape[0]):
            if n[j] == 0:
                # Set Gaussian value to avoid placement by random set
                P[j] = 0.0

                # Set Gaussian center extremely positive outside of range to avoid placement by default set
                B1[j] += 1e9

            # Check to see if most negative phase is past due to have all pixels
            elif P[j] < 1e-6 and j == np.argmin(B1 - x[k]):
                past_due = True

        # Sum of all Gaussians at this x position
        Psum = np.sum(P)

        # Random number for mixing of layers
        r = np.random.rand()

        # If x position is outside every Gaussian peak (default set)
        if Psum < 1e-6 or past_due:
            # Calculate the index of the most negative phase
            j = np.argmin(B1 - x[k])
            # Set pixel value
            L[l] = colors[j]
            # Subtract pixel for that phase
            n[j] -= 1
        # If x position is inside at least one Gaussian peak (random set)
        else:
            p = 0.0
            placed = False
            j = -1
            # Loop until pixel is set
            while not placed:
                j += 1
                if j == P.shape[0]:
                    print('--p:{}---------------Psum:{}-----------P[]:{}--------r:{}----------------------{}'.format(p, Psum, P, r))                
                p += P[j]
                # If random number is less than relative probability for that phase
                if r - p / Psum < 1e-6:
                    # Set pixel value
                    L[l] = colors[j]
                    # Subtract pixel for that phase
                    n[j] -= 1
                    # End loop
                    placed = True
    
    return L

def mix(A, B, C, D, Spol, Lpol, S, mixing):
    # Initialize time variable such that Gaussians have normalized area
    C = np.max([C, 1e-10])
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Time of fully mixed solution
    tmix = -1.0 * np.log(Cmix * np.sqrt(2.0 * np.pi))

    # Check if fully mixed already
    if t + mixing < tmix:
        mixing = tmix - t

    t += mixing

    # Reset layer positions to 0 and mix 
    # Update shift of Gaussian peaks
    B *= 0
    for i in range(B.shape[0]):
        for j in range(0, i):
            B[j] -= (D[j] - D[i]) * (t - tmix)
        for j in range(i+1, B.shape[0]):
            B[j] -= (D[j] - D[i]) * (t - tmix)

    Sts = np.zeros(S.shape)
    Scur = np.copy(S)
    # only do the calculation if there are two or more solvent
    if (len(A) < 2):
        C = np.exp(-1.0 * t) / np.sqrt(2.0 * np.pi)
        return B, C, S, Sts

    for i in range(S.shape[0]):
        # Calculate the relative and weighted polarity terms
        Ldif = 0
        Ldif0 = 0
        for j in range(Lpol.shape[0]):
            Ldif += abs(Spol[i] - Lpol[j])
            Ldif0 += A[j] * abs(Spol[i] - Lpol[j])

        # Calculate total amount of solute
        Ssum = np.sum(Scur[i])

        # Calculate constant
        c = 1 / (1 - (Ldif0) / (np.sum(A) * Ldif))

        # Calculate the ideal amount of solute i in each phase
        # Check conditions that this adds to Ssum
        St = (Ssum * A / np.sum(A)) * (np.exp(tmix - t) + c * (1 - np.exp(tmix - t)) * (1 - (abs(Spol[i] - Lpol) / Ldif)))
        Sts[i] = np.copy(St)

    # Update amount of solute i in each phase
    for i in range(S.shape[0]):
        # Calculate the relative and weighted polarity terms
        Ldif = 0
        Ldif0 = 0
        for j in range(Lpol.shape[0]):
            Ldif += abs(Spol[i] - Lpol[j])
            Ldif0 += A[j] * abs(Spol[i] - Lpol[j])

        # Calculate total amount of solute
        Ssum = np.sum(Scur[i])

        # Calculate constant
        c = 1 / (1 - (Ldif0) / (np.sum(A) * Ldif))

        # Calculate the ideal amount of solute i in each phase for the previous time step
        St0 = (Ssum * A / np.sum(A)) * (np.exp(tmix - t + mixing) + c * (1 - np.exp(tmix - t + mixing)) * (1 - (abs(Spol[i] - Lpol) / Ldif)))

        if abs(t - mixing - tmix) > 1e-9:
            S[i] = Sts[i] + 0.5 * ((1 - abs(mixing) / mixing) * (S[i] - St0) * (t - tmix) / (t - mixing - tmix) + (1 + abs(mixing) / mixing) * (S[i] - St0))
        else:
            S[i] = Sts[i]

    # Update the varience of each Gaussian peak
    C = np.exp(-1.0 * t) / np.sqrt(2.0 * np.pi)

    return B, C, S, Sts

