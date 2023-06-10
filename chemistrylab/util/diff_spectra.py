"""
Please pay attention we have defined the spectral peak locations in terms of the their
location in cm^-1
"""
import numpy as np
import copy


def convert_inverse_cm_to_nm(params: list):
    params = copy.deepcopy(params)
    for i in range(len(params)):
        if params[i] is not None:
            for k in range(len(params[i])):
                params[i][k][1] = 1/params[i][k][1] * 1e7
    return params


def convert_nm_to_inverse_cm(params: list):
    params = copy.deepcopy(params)
    for i in range(len(params)):
        if params[i] is not None:
            for k in range(len(params[i])):
                params[i][k][1] = 1/params[i][k][1] * 1e-7
    return params


S_1_chlorohexane = np.zeros((7, 3))
S_1_chlorohexane[0, 0] = 0.75
S_1_chlorohexane[0, 1] = 2950
S_1_chlorohexane[0, 2] = 0.001

S_1_chlorohexane[1, 0] = 0.832
S_1_chlorohexane[1, 1] = 2940
S_1_chlorohexane[1, 2] = 0.002

S_1_chlorohexane[2, 0] = 0.388
S_1_chlorohexane[2, 1] = 2880
S_1_chlorohexane[2, 2] = 0.009

S_1_chlorohexane[3, 0] = 0.133
S_1_chlorohexane[3, 1] = 1450
S_1_chlorohexane[3, 2] = 0.010

S_1_chlorohexane[4, 0] = 0.09
S_1_chlorohexane[4, 1] = 1300
S_1_chlorohexane[4, 2] = 0.015

S_1_chlorohexane[5, 0] = 0.108
S_1_chlorohexane[5, 1] = 735
S_1_chlorohexane[5, 2] = 0.01

S_1_chlorohexane[6, 0] = 0.062
S_1_chlorohexane[6, 1] = 660
S_1_chlorohexane[6, 2] = 0.008


S_2_chlorohexane = np.zeros((18, 3))
S_2_chlorohexane[0, 0] = 0.93
S_2_chlorohexane[0, 1] = 2950
S_2_chlorohexane[0, 2] = 0.0056

S_2_chlorohexane[1, 0] = 0.9
S_2_chlorohexane[1, 1] = 2850
S_2_chlorohexane[1, 2] = 0.0017

S_2_chlorohexane[2, 0] = 0.77
S_2_chlorohexane[2, 1] = 1440
S_2_chlorohexane[2, 2] = 0.0018

S_2_chlorohexane[3, 0] = 0.79
S_2_chlorohexane[3, 1] = 1390
S_2_chlorohexane[3, 2] = 0.0088

S_2_chlorohexane[4, 0] = 0.52
S_2_chlorohexane[4, 1] = 1310
S_2_chlorohexane[4, 2] = 0.005

S_2_chlorohexane[5, 0] = 0.7
S_2_chlorohexane[5, 1] = 1260
S_2_chlorohexane[5, 2] = 0.006

S_2_chlorohexane[6, 0] = 0.52
S_2_chlorohexane[6, 1] = 1200
S_2_chlorohexane[6, 2] = 0.007

S_2_chlorohexane[7, 0] = 0.59
S_2_chlorohexane[7, 1] = 1160
S_2_chlorohexane[7, 2] = 0.003

S_2_chlorohexane[8, 0] = 0.52
S_2_chlorohexane[8, 1] = 1100
S_2_chlorohexane[8, 2] = 0.007

S_2_chlorohexane[9, 0] = 0.63
S_2_chlorohexane[9, 1] = 1040
S_2_chlorohexane[9, 2] = 0.004

S_2_chlorohexane[10, 0] = 0.63
S_2_chlorohexane[10, 1] = 1010
S_2_chlorohexane[10, 2] = 0.0045

S_2_chlorohexane[11, 0] = 0.7
S_2_chlorohexane[11, 1] = 980
S_2_chlorohexane[11, 2] = 0.007

S_2_chlorohexane[12, 0] = 0.5
S_2_chlorohexane[12, 1] = 900
S_2_chlorohexane[12, 2] = 0.0061

S_2_chlorohexane[13, 0] = 0.2
S_2_chlorohexane[13, 1] = 860
S_2_chlorohexane[13, 2] = 0.005

S_2_chlorohexane[14, 0] = 0.3
S_2_chlorohexane[14, 1] = 830
S_2_chlorohexane[14, 2] = 0.0014

S_2_chlorohexane[15, 0] = 0.55
S_2_chlorohexane[15, 1] = 780
S_2_chlorohexane[15, 2] = 0.007

S_2_chlorohexane[16, 0] = 0.65
S_2_chlorohexane[16, 1] = 720
S_2_chlorohexane[16, 2] = 0.008

S_2_chlorohexane[17, 0] = 0.80
S_2_chlorohexane[17, 1] = 620
S_2_chlorohexane[17, 2] = 0.006


S_3_chlorohexane = np.zeros((3, 3))
S_3_chlorohexane[0, 0] = 0.0293
S_3_chlorohexane[0, 1] = 2966
S_3_chlorohexane[0, 2] = 0.006

S_3_chlorohexane[1, 0] = 0.0118
S_3_chlorohexane[1, 1] = 2889
S_3_chlorohexane[1, 2] = 0.002

S_3_chlorohexane[2, 0] = 0.0048
S_3_chlorohexane[2, 1] = 1462
S_3_chlorohexane[2, 2] = 0.006

### --------- Fict Materials --------- ###

S_A = np.array([[9.43904295e-01, 2.41678874e+03, 9.95871813e-03],
                [9.93508858e-01, 1.61688257e+03, 1.95518549e-02],
                [5.04060614e-01, 4.12622244e+02, 2.64196495e-03],
                [1.57465892e-01, 7.07813158e+02, 7.55995404e-04]])

S_B = np.array([[1.85140447e-01, 2.87863429e+03, 7.13364826e-03],
                [9.18588276e-01, 7.58568779e+02, 7.38481312e-03],
                [7.38631244e-02, 1.55234830e+03, 3.13263193e-02]])

S_C = np.array([[5.28979037e-01, 2.99592773e+03, 3.29373665e-02],
                [9.43872169e-01, 2.37783762e+03, 6.15995397e-03],
                [6.13353671e-01, 2.98201571e+03, 9.85076476e-03]])

S_D = np.array([[1.90579725e-01, 2.55963467e+03, 3.49184581e-03]])

S_E = np.array([[1.81834079e-01, 1.26477724e+03, 1.36971571e-03],
                [6.54988735e-02, 7.62889243e+02, 1.89455691e-03]])

S_F = np.array([[4.20758409e-01, 2.46074517e+02, 5.57671503e-02]])

S_G = np.array([[5.03802045e-01, 2.21412634e+03, 1.33214062e-03],
                [9.30694466e-01, 5.43742572e+02, 9.58193826e-03],
                [8.58834516e-01, 1.46912516e+03, 3.04494718e-03]])

S_H = np.array([[9.77456880e-01, 2.61364890e+03, 8.92133157e-03],
                [9.60887633e-01, 2.95398827e+03, 8.55542947e-02]])

S_I = np.array([[1.03611024e-01, 1.06505255e+03, 1.63771049e-03]])

def normalize_spectra(params, ir_range):
    spectras = copy.deepcopy(params)
    min = ir_range[0]
    wave_range = abs(ir_range[1] - ir_range[0])
    for i in range(len(spectras)):
        if spectras[i] is not None:
            for k in range(len(spectras[i])):
                spectras[i][k][1] = abs(spectras[i][k][1] - min) / wave_range
    return spectras

# plot these triple peak spectra
def plot_mix_peak():
    wave_min = 2000
    wave_max = 20000
    triple_peak = [S_2_chlorohexane]
    triple_peak = convert_inverse_cm_to_nm(triple_peak)
    triple_peak = normalize_spectra(triple_peak, (wave_min, wave_max))
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
    import matplotlib.pyplot as plt
    plot_mix_peak()