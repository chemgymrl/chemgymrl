import numpy as np

V = 0.001 # Volumre of reaction vessel (m**3)
T = 300.0 # Temperature of system (K)
R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)

A = 1.0 # Mol of A
B = 0.5 # Mol of B
C = 0.2 # Mol of C

# Array of mols of each species
n = np.zeros(3, dtype=np.float32)
n[0] = A
n[1] = B
n[2] = C

P = np.zeros(n.shape[0], dtype=np.float32) # Initial partial pressures (kPa)

# Each species independantly contributes to total pressure calculated by ideal gas law
for i in range(P.shape[0]):
    P[i] += n[i] * R * T / V

P_total = np.sum(P)

print('Pressure of A: %.3f kPa\nPressure of B: %.3f kPa\nPressure of C: %.3f kPa\nPressure of system: %.3f kPa' % (P[0], P[1], P[2], P_total))
