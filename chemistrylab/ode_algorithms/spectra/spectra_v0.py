import numpy as np
import matplotlib.pyplot as plt

wave_min = 200 # Minimum wavelength (nm)
wave_max = 800 # Maximum wavelength (nm)

A = 1.0 # Mol of A
B = 0.5 # Mol of B
C = 0.2 # Mol of C

# Array of mols of each species
n = np.zeros(3, dtype=np.float32)
n[0] = A
n[1] = B
n[2] = C

# Parameters to generate up to 3 Gaussian peaks per species
params = np.zeros((n.shape[0], 3, 3), dtype=np.float32)

# Parameters for species A peaks
params[0, 0, 0] = 0.2 # Maximum height of peak 0
params[0, 0, 1] = 0.3 # Position of peak 0 on x-axis
params[0, 0, 2] = 0.02 # Variance of peak 0

params[0, 1, 0] = 0.6 # Maximum height of peak 1
params[0, 1, 1] = 0.5 # Position of peak 1 on x-axis
params[0, 1, 2] = 0.02 # Variance of peak 1

params[0, 2, 0] = 0.0 # Maximum height of peak 2
params[0, 2, 1] = -1.0 # Position of peak 2 on x-axis
params[0, 2, 2] = 1.0 # Variance of peak 2

# Parameters for species B peaks
params[1, 0, 0] = 0.6 # Maximum height of peak 0
params[1, 0, 1] = 0.7 # Position of peak 0 on x-axis
params[1, 0, 2] = 0.01 # Variance of peak 0

params[1, 1, 0] = 0.3 # Maximum height of peak 1
params[1, 1, 1] = 0.65 # Position of peak 1 on x-axis
params[1, 1, 2] = 0.01 # Variance of peak 1

params[1, 2, 0] = 0.3 # Maximum height of peak 2
params[1, 2, 1] = 0.75 # Position of peak 2 on x-axis
params[1, 2, 2] = 0.01 # Variance of peak 2

# Parameters for species C peaks
params[2, 0, 0] = 0.9 # Maximum height of peak 0
params[2, 0, 1] = 0.1 # Position of peak 0 on x-axis
params[2, 0, 2] = 0.04 # Variance of peak 0

params[2, 1, 0] = 0.0 # Maximum height of peak 1
params[2, 1, 1] = -1.0 # Position of peak 1 on x-axis
params[2, 1, 2] = 1.0 # Variance of peak 1

params[2, 2, 0] = 0.0 # Maximum height of peak 2
params[2, 2, 1] = -1.0 # Position of peak 2 on x-axis
params[2, 2, 2] = 1.0 # Variance of peak 2

# Initialize array for wavelength[0, 1] and absorbance
x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
absorb = np.zeros(x.shape[0], dtype=np.float32)

for i in range(params.shape[0]):
    for j in range(params.shape[1]):
        for k in range(x.shape[0]):
            absorb[k] += n[i] * params[i, j, 0] * np.exp(-0.5 * ((x[k] - params[i, j, 1]) / params[i, j, 2]) ** 2.0)

# Maximum possible absorbance at any wavelength is 1.0
absorb = np.clip(absorb, 0.0, 1.0)

# Wavelength array for plotting
wave = np.linspace(wave_min, wave_max, x.shape[0], endpoint=True, dtype=np.float32)

plt.figure()
plt.plot(wave, absorb)
# Creating peak labels for demo purposes and 'full' render mode
plt.scatter(params[0, :, 1] * (wave_max - wave_min) + wave_min, n[0] * params[0, :, 0], label='A')
plt.scatter(params[1, :, 1] * (wave_max - wave_min) + wave_min, n[1] * params[1, :, 0], label='B')
plt.scatter(params[2, :, 1] * (wave_max - wave_min) + wave_min, n[2] * params[2, :, 0], label='C')
plt.xlim([wave_min, wave_max])
plt.ylim([0, 1])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.legend()
plt.show()
plt.close()
