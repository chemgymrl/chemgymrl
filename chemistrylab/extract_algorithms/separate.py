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
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Number of pixels available for each phase
    sum_A = 1.0 if np.sum(A) == 0 else np.sum(A)
    n = ((A / sum_A) * L.shape[0]).astype(int)

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
                    print('--p:{}---------------Psum:{}-----------P[]:{}--------r:{}----------------------'.format(p, Psum, P, r))
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
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Time of fully mixed solution
    tmix = -1.0 * np.log(Cmix * np.sqrt(2.0 * np.pi))

    # Check if fully mixed already
    if t + mixing < tmix:
        mixing = tmix - t

    t += mixing

    # Update shift of Gaussian peaks
    for i in range(B.shape[0]):
        for j in range(0, i):
            B[j] -= (D[j] - D[i]) * mixing
        for j in range(i+1, B.shape[0]):
            B[j] -= (D[j] - D[i]) * mixing

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
            #S[i] = Scur[i]
        else:
            S[i] = Sts[i]
            #S[i] = Scur[i]

    # Update the varience of each Gaussian peak
    C = np.exp(-1.0 * t) / np.sqrt(2.0 * np.pi)

    return B, C, S, Sts

'''
# Array for Gaussians at each time step
Ps = np.zeros((100, A.shape[0], x.shape[0]), dtype=np.float32)

# Array for layers at each time step
Ls = np.zeros((Ps.shape[0], 100), dtype=np.float32) + colors[-1]

# Array for solutes at each time step
Ss = np.zeros((Ps.shape[0], S.shape[0], S.shape[1]), dtype=np.float32) + colors[-1]
Sts = np.zeros((Ps.shape[0], S.shape[0], S.shape[1]), dtype=np.float32) + colors[-1]

B, C, S, St = mix(A[:-1], B, C, D, Spol, Lpol, S, 0.0)

# Loop over each time step for positive dt
for i in range(Ps.shape[0]//2):
    # Initialize time variable such that Gaussians have normalized area
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Loop over each x position
    for j in range(Ps.shape[1]):
        # Loop over each phase
        for k in range(Ps.shape[2]):
            # Calculate the value of the Gaussian for that phase and x position
            Ps[i, j, k] = A[j] * np.exp(-1.0 * (((x[k] - B[j]) / (2.0 * C)) ** 2) + t)

    Ls[i] = map_to_state(A, B, C, x)
    B, C, S, St = mix(A[:-1], B, C, D, Spol, Lpol, S, dt)
    Ss[i] = np.copy(S)
    Sts[i] = np.copy(St)

S = np.array([[0.75, 0.25], [0.05, 0.95]])

# Loop over each time step for negative dt
for i in range(Ps.shape[0]//2):
    # Initialize time variable such that Gaussians have normalized area
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Loop over each x position
    for j in range(Ps.shape[1]):
        # Loop over each phase
        for k in range(Ps.shape[2]):
            # Calculate the value of the Gaussian for that phase and x position
            Ps[i, j, k] = A[j] * np.exp(-1.0 * (((x[k] - B[j]) / (2.0 * C)) ** 2) + t)

    Ls[i] = map_to_state(A, B, C, x)
    B, C, S, St = mix(A[:-1], B, C, D, Spol, Lpol, S, -1.0*dt)
    Ss[i] = np.copy(S)
    Sts[i] = np.copy(St)

for i in range(Ps.shape[0]//2, Ps.shape[0]):
    # Initialize time variable such that Gaussians have normalized area
    t = -1.0 * np.log(C * np.sqrt(2.0 * np.pi))

    # Loop over each x position
    for j in range(Ps.shape[1]):
        # Loop over each phase
        for k in range(Ps.shape[2]):
            # Calculate the value of the Gaussian for that phase and x position
            Ps[i, j, k] = A[j] * np.exp(-1.0 * (((x[k] - B[j]) / (2.0 * C)) ** 2) + t)

    Ls[i] = map_to_state(A, B, C, x)
    B, C, S, St = mix(A[:-1], B, C, D, Spol, Lpol, S, dt)
    Ss[i] = np.copy(S)
    Sts[i] = np.copy(St)

B, C, S, St = mix(A[:-1], B, C, D, Spol, Lpol, S, -1.0*dt)

Ls = np.reshape(Ls, (Ls.shape[0], Ls.shape[1], 1))

# Plot Gaussian peaks and phase layers for each time step
for i in range(Ps.shape[0]):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [4, 4, 1]})
    for j in range(Ps.shape[1]):
        ax[0].plot(x, Ps[i, j], label=labels[j])
    ax[0].set_xlim([x[0], x[-1]])
    ax[0].set_ylim([0, 1])
    ax[0].set_xlabel('Separation')
    ax[0].legend()
    ax[1].bar([0, 1, 2, 3], Ss[i].flatten())
    ax[1].set_xticks([0, 1, 2, 3])
    ax[1].set_xticklabels(['S1 Water', 'S1 Oil', 'S2 Water', 'S2 Oil'])
    ax[1].set_ylim([0, 1])
    mappable = ax[2].pcolormesh(Ls[i], vmin=0, vmax=1, cmap=cmocean.cm.delta)
    ax[2].set_xticks([])
    ax[2].set_ylabel('Height')
    fig.colorbar(mappable)
    plt.savefig('./figures/ex_%.3i.png' % (i), dpi=256)
    plt.close()

x = np.linspace(0, 1, Ss.shape[0])
labels1 = ['1', '2']
labels2 = ['Water', 'Oil', 'Other']
colour = ['b', 'r', 'g', 'k', 'c', 'm']
k = 0

plt.figure()
for i in range(Ss.shape[1]):
    for j in range(Ss.shape[2]):
        plt.plot(x, Ss[:, i, j], c=colour[k], label=labels1[i] + ' ' + labels2[j])
        plt.plot(x, Sts[:, i, j], c=colour[k], ls='--', lw=2.0, label=labels1[i] + ' ' + labels2[j] + ' True')
        k += 1

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Mixing Time')
plt.ylabel('Amount of Solute')
plt.xticks([0, 0.5, 1], ['Settled', 'Mixed', 'Settled'])
plt.legend()
plt.show()
plt.close()
'''
