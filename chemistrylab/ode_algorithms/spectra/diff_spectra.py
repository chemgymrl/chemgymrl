import numpy as np
import matplotlib.pyplot as plt

wave_min = 200 # Minimum wavelength (nm)
wave_max = 800 # Maximum wavelength (nm)

# 1 peak spectra, same shape but location go from 0.1 to 0.9
S_1 = np.zeros((1, 3), dtype=np.float32)
S_2 = np.zeros((1, 3), dtype=np.float32)
S_3 = np.zeros((1, 3), dtype=np.float32)
S_4 = np.zeros((1, 3), dtype=np.float32)
S_5 = np.zeros((1, 3), dtype=np.float32)
S_6 = np.zeros((1, 3), dtype=np.float32)
S_7 = np.zeros((1, 3), dtype=np.float32)
S_8 = np.zeros((1, 3), dtype=np.float32)
S_9 = np.zeros((1, 3), dtype=np.float32)
# Spectra_1 
S_1[0, 0] = 0.8 # Maximum height of peak 0
S_1[0, 1] = 0.1 # Position of peak 0 on x-axis
S_1[0, 2] = 0.015 # Variance of peak 0
# Spectra_2
S_2[0, 0] = 0.8 # Maximum height of peak 0
S_2[0, 1] = 0.2 # Position of peak 0 on x-axis
S_2[0, 2] = 0.015 # Variance of peak 0
# Spectra_3
S_3[0, 0] = 0.8 # Maximum height of peak 0
S_3[0, 1] = 0.3 # Position of peak 0 on x-axis
S_3[0, 2] = 0.015 # Variance of peak 0
# Spectra_4
S_4[0, 0] = 0.8 # Maximum height of peak 0
S_4[0, 1] = 0.4 # Position of peak 0 on x-axis
S_4[0, 2] = 0.015 # Variance of peak 0
# Spectra_5
S_5[0, 0] = 0.8 # Maximum height of peak 0
S_5[0, 1] = 0.5 # Position of peak 0 on x-axis
S_5[0, 2] = 0.015 # Variance of peak 0
# Spectra_6
S_6[0, 0] = 0.8 # Maximum height of peak 0
S_6[0, 1] = 0.6 # Position of peak 0 on x-axis
S_6[0, 2] = 0.015 # Variance of peak 0
# Spectra_7
S_7[0, 0] = 0.8 # Maximum height of peak 0
S_7[0, 1] = 0.7 # Position of peak 0 on x-axis
S_7[0, 2] = 0.015 # Variance of peak 0
# Spectra_8
S_8[0, 0] = 0.8 # Maximum height of peak 0
S_8[0, 1] = 0.8 # Position of peak 0 on x-axis
S_8[0, 2] = 0.015 # Variance of peak 0
# Spectra_9
S_9[0, 0] = 0.8 # Maximum height of peak 0
S_9[0, 1] = 0.9 # Position of peak 0 on x-axis
S_9[0, 2] = 0.015 # Variance of peak 0

# spectra with 3 peak, same shape but location diff
S_1_3 = np.zeros((3, 3), dtype=np.float32)
S_2_3 = np.zeros((3, 3), dtype=np.float32)
S_3_3 = np.zeros((3, 3), dtype=np.float32)
# Spectra_1_3 # 3 peak
S_1_3[0, 0] = 0.5 # Maximum height of peak 0
S_1_3[0, 1] = 0.0 # Position of peak 0 on x-axis
S_1_3[0, 2] = 0.015 # Variance of peak 0

S_1_3[1, 0] = 0.9 # Maximum height of peak 1
S_1_3[1, 1] = 0.1 # Position of peak 1 on x-axis
S_1_3[1, 2] = 0.015 # Variance of peak 1

S_1_3[2, 0] = 0.5 # Maximum height of peak 2
S_1_3[2, 1] = 0.2 # Position of peak 2 on x-axis
S_1_3[2, 2] = 0.015 # Variance of peak 0
# Spectra_2_3 # 3 peak
S_2_3[0, 0] = 0.5 # Maximum height of peak 0
S_2_3[0, 1] = 0.35 # Position of peak 0 on x-axis
S_2_3[0, 2] = 0.015 # Variance of peak 0

S_2_3[1, 0] = 0.9 # Maximum height of peak 1
S_2_3[1, 1] = 0.45 # Position of peak 1 on x-axis
S_2_3[1, 2] = 0.015 # Variance of peak 1

S_2_3[2, 0] = 0.5 # Maximum height of peak 2
S_2_3[2, 1] = 0.55 # Position of peak 2 on x-axis
S_2_3[2, 2] = 0.015 # Variance of peak 0
# Spectra_3_3 # 3 peak
S_3_3[0, 0] = 0.5 # Maximum height of peak 0
S_3_3[0, 1] = 0.65 # Position of peak 0 on x-axis
S_3_3[0, 2] = 0.015 # Variance of peak 0

S_3_3[1, 0] = 0.9 # Maximum height of peak 1
S_3_3[1, 1] = 0.75 # Position of peak 1 on x-axis
S_3_3[1, 2] = 0.015 # Variance of peak 1

S_3_3[2, 0] = 0.5 # Maximum height of peak 2
S_3_3[2, 1] = 0.85 # Position of peak 2 on x-axis
S_3_3[2, 2] = 0.015 # Variance of peak 0


# plot these single peak spectra
def plot_single_peak():
    single_peak = [S_1, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9]
    # Initialize array for wavelength[0, 1] and absorbance
    x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
    absorb = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(9): # each spectra
            for j in range(x.shape[0]):
                absorb[j] += single_peak[i][0, 0] * np.exp(-0.5 * ((x[j] - single_peak[i][0, 1]) / single_peak[i][0, 2]) ** 2.0)
                
    wave = np.linspace(wave_min, wave_max, x.shape[0], endpoint=True, dtype=np.float32)
    plt.figure()
    plt.plot(wave, absorb)
    # label
    for i in range(9):
        label = "S_" + str(i+1)
        plt.scatter(single_peak[i][:, 1] * (wave_max - wave_min) + wave_min, single_peak[i][:, 0], label=label)


    plt.xlim([wave_min, wave_max])
    plt.ylim([0, 1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()
    plt.close()

# plot these triple peak spectra
def plot_triple_peak():
    triple_peak = [S_1_3, S_2_3, S_3_3]
    # Initialize array for wavelength[0, 1] and absorbance
    x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
    absorb = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(3): # each spectra
        for j in range(triple_peak[i].shape[0]):
            for k in range(x.shape[0]):
                absorb[k] += triple_peak[i][j, 0] * np.exp(-0.5 * ((x[k] - triple_peak[i][j, 1]) / triple_peak[i][j, 2]) ** 2.0)
                
    wave = np.linspace(wave_min, wave_max, x.shape[0], endpoint=True, dtype=np.float32)
    plt.figure()
    plt.plot(wave, absorb)
    # label
    for i in range(3):
        label = "S_" + str(i+1) + "_3"
        plt.scatter(triple_peak[i][:, 1] * (wave_max - wave_min) + wave_min, triple_peak[i][:, 0], label=label)


    plt.xlim([wave_min, wave_max])
    plt.ylim([0, 1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()
    plt.close()

# plot these triple peak spectra
def plot_mix_peak():
    triple_peak = [S_1, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9, S_1_3, S_2_3, S_3_3]
    # Initialize array for wavelength[0, 1] and absorbance
    x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
    wave = np.linspace(wave_min, wave_max, x.shape[0], endpoint=True, dtype=np.float32)
    absorb = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(12): # each spectra
        for j in range(triple_peak[i].shape[0]):
            for k in range(x.shape[0]):
                absorb[k] += triple_peak[i][j, 0] * np.exp(-0.5 * ((x[k] - triple_peak[i][j, 1]) / triple_peak[i][j, 2]) ** 2.0)
    plt.figure()
    # plot the overlap spectra in solid line
    plt.plot(wave, absorb)
    # plot each spectra in dash line
    for i in range(12):
        dash_absorb = np.zeros(x.shape[0], dtype=np.float32)
        for j in range(triple_peak[i].shape[0]):
            for k in range(x.shape[0]):
                dash_absorb[k] += triple_peak[i][j, 0] * np.exp(-0.5 * ((x[k] - triple_peak[i][j, 1]) / triple_peak[i][j, 2]) ** 2.0)
        plt.plot(wave, dash_absorb, linestyle='dashed')
    # label
    for i in range(12):
        label = "S_" + str(i+1)
        plt.scatter(triple_peak[i][:, 1] * (wave_max - wave_min) + wave_min, triple_peak[i][:, 0], label=label)


    plt.xlim([wave_min, wave_max])
    plt.ylim([0, 1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()
    plt.close()

