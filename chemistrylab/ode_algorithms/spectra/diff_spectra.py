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

S_1_chlorohexane = np.zeros((7, 3))
S_1_chlorohexane[0, 0] = 0.75
S_1_chlorohexane[0, 1] = 0.157
S_1_chlorohexane[0, 2] = 0.0012

S_1_chlorohexane[1, 0] = 0.832
S_1_chlorohexane[1, 1] = 0.16
S_1_chlorohexane[1, 2] = 0.0025

S_1_chlorohexane[2, 0] = 0.388
S_1_chlorohexane[2, 1] = 0.177
S_1_chlorohexane[2, 2] = 0.009

S_1_chlorohexane[3, 0] = 0.133
S_1_chlorohexane[3, 1] = 0.584
S_1_chlorohexane[3, 2] = 0.010

S_1_chlorohexane[4, 0] = 0.09
S_1_chlorohexane[4, 1] = 0.63
S_1_chlorohexane[4, 2] = 0.015

S_1_chlorohexane[5, 0] = 0.108
S_1_chlorohexane[5, 1] = 0.790
S_1_chlorohexane[5, 2] = 0.01

S_1_chlorohexane[6, 0] = 0.062
S_1_chlorohexane[6, 1] = 0.813
S_1_chlorohexane[6, 2] = 0.008


S_2_chlorohexane = np.zeros((18, 3))
S_2_chlorohexane[0, 0] = 0.93
S_2_chlorohexane[0, 1] = 1-2950/3500
S_2_chlorohexane[0, 2] = 0.010

S_2_chlorohexane[1, 0] = 0.9
S_2_chlorohexane[1, 1] = 1-2850/3500
S_2_chlorohexane[1, 2] = 0.009

S_2_chlorohexane[2, 0] = 0.87
S_2_chlorohexane[2, 1] = 1-1440/3500
S_2_chlorohexane[2, 2] = 0.0018

S_2_chlorohexane[3, 0] = 0.89
S_2_chlorohexane[3, 1] = 1-1390/3500
S_2_chlorohexane[3, 2] = 0.001

S_2_chlorohexane[4, 0] = 0.6
S_2_chlorohexane[4, 1] = 1-1310/3500
S_2_chlorohexane[4, 2] = 0.01

S_2_chlorohexane[5, 0] = 0.8
S_2_chlorohexane[5, 1] = 1-1260/3500
S_2_chlorohexane[5, 2] = 0.002

S_2_chlorohexane[6, 0] = 0.52
S_2_chlorohexane[6, 1] = 1-1200/3500
S_2_chlorohexane[6, 2] = 0.007

S_2_chlorohexane[7, 0] = 0.59
S_2_chlorohexane[7, 1] = 1-1160/3500
S_2_chlorohexane[7, 2] = 0.001

S_2_chlorohexane[8, 0] = 0.52
S_2_chlorohexane[8, 1] = 1-1100/3500
S_2_chlorohexane[8, 2] = 0.004

S_2_chlorohexane[9, 0] = 0.63
S_2_chlorohexane[9, 1] = 1-1040/3500
S_2_chlorohexane[9, 2] = 0.008

S_2_chlorohexane[10, 0] = 0.63
S_2_chlorohexane[10, 1] = 1-1010/3500
S_2_chlorohexane[10, 2] = 0.09

S_2_chlorohexane[11, 0] = 0.7
S_2_chlorohexane[11, 1] = 1-980/3500
S_2_chlorohexane[11, 2] = 0.005

S_2_chlorohexane[12, 0] = 0.5
S_2_chlorohexane[12, 1] = 1-900/3500
S_2_chlorohexane[12, 2] = 0.0061

S_2_chlorohexane[13, 0] = 0.2
S_2_chlorohexane[13, 1] = 1-860/3500
S_2_chlorohexane[13, 2] = 0.0012

S_2_chlorohexane[14, 0] = 0.3
S_2_chlorohexane[14, 1] = 1-830/3500
S_2_chlorohexane[14, 2] = 0.0014

S_2_chlorohexane[15, 0] = 0.65
S_2_chlorohexane[15, 1] = 1-780/3500
S_2_chlorohexane[15, 2] = 0.0052

S_2_chlorohexane[16, 0] = 0.73
S_2_chlorohexane[16, 1] = 1-720/3500
S_2_chlorohexane[16, 2] = 0.0014

S_2_chlorohexane[10, 0] = 0.86
S_2_chlorohexane[10, 1] = 1-620/3500
S_2_chlorohexane[10, 2] = 0.006


S_3_chlorohexane = np.zeros((3, 3))
S_3_chlorohexane[0, 0] = 0.0293
S_3_chlorohexane[0, 1] = 1-2966/3500
S_3_chlorohexane[0, 2] = 0.006

S_3_chlorohexane[1, 0] = 0.0118
S_3_chlorohexane[1, 1] = 1-2889/3500
S_3_chlorohexane[1, 2] = 0.002

S_3_chlorohexane[2, 0] = 0.0048
S_3_chlorohexane[2, 1] = 1-1462/3500
S_3_chlorohexane[2, 2] = 0.006

### ------------ dodecane -------------- ###
S_dodecane = np.zeros((6,3))
S_dodecane[0,0] = 0.403
S_dodecane[0,1] = 0.157
S_dodecane[0,2] = 0.001

S_dodecane[1,0] = 0.95
S_dodecane[1,1] = 0.16
S_dodecane[1,2] = 0.003

S_dodecane[2,0] = 0.36
S_dodecane[2,1] = 0.180
S_dodecane[2,2] = 0.008

S_dodecane[3,0] = 0.069
S_dodecane[3,1] = 0.588
S_dodecane[3,2] = 0.012

S_dodecane[4,0] = 0.025
S_dodecane[4,1] = 0.615
S_dodecane[4,2] = 0.007

S_dodecane[5,0] = 0.016
S_dodecane[5,1] = 0.8
S_dodecane[5,2] = 0.007

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
    triple_peak = [S_3_chlorohexane]
    # Initialize array for wavelength[0, 1] and absorbance
    x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
    wave = np.linspace(wave_min, wave_max, x.shape[0], endpoint=True, dtype=np.float32)
    absorb = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(len(triple_peak)): # each spectra
        for j in range(triple_peak[i].shape[0]):
            for k in range(x.shape[0]):
                absorb[k] += triple_peak[i][j, 0] * np.exp(-0.5 * ((x[k] - triple_peak[i][j, 1]) / triple_peak[i][j, 2]) ** 2.0)
    plt.figure()
    # plot the overlap spectra in solid line
    plt.plot(wave, absorb)
    # plot each spectra in dash line
    for i in range(len(triple_peak)):
        dash_absorb = np.zeros(x.shape[0], dtype=np.float32)
        for j in range(triple_peak[i].shape[0]):
            for k in range(x.shape[0]):
                dash_absorb[k] += triple_peak[i][j, 0] * np.exp(-0.5 * ((x[k] - triple_peak[i][j, 1]) / triple_peak[i][j, 2]) ** 2.0)
        plt.plot(wave, dash_absorb, linestyle='dashed')
    # label
    for i in range(len(triple_peak)):
        label = "S_" + str(i+1)
        plt.scatter(triple_peak[i][:, 1] * (wave_max - wave_min) + wave_min, triple_peak[i][:, 0], label=label)


    plt.xlim([wave_min, wave_max])
    plt.ylim([0, 1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_mix_peak()
