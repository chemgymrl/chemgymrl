import numpy as np
import matplotlib.pyplot as plt
from chemistrylab.ode_algorithms.spectra import diff_spectra as spec

# Reactions
# 1) A + B + C -> E
# 2) A + D -> F
# 3) B + D -> G
# 4) C + D -> H
# 5) F + G + H -> |I|

R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)
# constant for k = e^(-E/RT) for each reaction
# 1) A + B + C -> E
A1 = 400.0
E1 = 50.0
# 2) A + D -> F
A2 = 0.55
E2 = 0.5
# 3) B + D -> G
A3 = 1.0
E3 = 3.0
# 4) C + D -> H
A4 = 0.7
E4 = 1.5
# 5) F + G + H -> |I|
A5 = 75.0
E5 = 40.0

class Reaction(object):
    def __init__(self, overlap = False):
        # A B C D is reactantn in hand
        self.initial_in_hand = np.array([1.0, 1.0, 1.0, 1.0]) # Amount of each reactant in hand (mols) avaliable
        self.nmax = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/3.0]) # The maximum of each chemical (mols)
        self.labels =  ['[A]', '[B]', '[C]', '[D]', '[E]', '[F]', '[G]', '[H]', '[I]']
        self.rate = np.zeros(5) # rate of each reaction
        self.max_mol = 4.0
        # Parameters to generate up to 3 Gaussian peaks per species
        self.params = []
        if overlap:
            # overlap spectra
            self.params.append(spec.S_3) # spectra for A
            self.params.append(spec.S_4) # spectra for B
            self.params.append(spec.S_5) # spectra for C
            self.params.append(spec.S_6) # spectra for D
            self.params.append(spec.S_7) # spectra for E
            self.params.append(spec.S_2_3) # spectra for F
            self.params.append(spec.S_3_3) # spectra for G
            self.params.append(spec.S_8) # spectra for H
            self.params.append(spec.S_9) # spectra for I
        else:
            # no overlap spectra
            self.params.append(spec.S_1) # spectra for A
            self.params.append(spec.S_2) # spectra for B
            self.params.append(spec.S_3) # spectra for C
            self.params.append(spec.S_4) # spectra for D
            self.params.append(spec.S_5) # spectra for E
            self.params.append(spec.S_6) # spectra for F
            self.params.append(spec.S_7) # spectra for G
            self.params.append(spec.S_8) # spectra for H
            self.params.append(spec.S_9) # spectra for I
        self.reset()

    def get_ni_label(self):
        return ['[A]', '[B]', '[C]', '[D]']

    def get_ni_num(self):
        num_list = []
        for i in range(self.cur_in_hand.shape[0]):
            num_list.append(self.cur_in_hand[i])
        return num_list

    # Reinitializes the reaction for reset purpose in main function
    def reset(self):
        self.cur_in_hand = 1.0 * self.initial_in_hand 
        self.n = np.zeros(self.nmax.shape[0], dtype=np.float32)
        #n[0] is A, n[1] is B, n[2] is C ......

    # update the concentration of each chemical for a time step under tempurature T
    def update(self, T, V, dt):
        C = self.get_concentration(V)
        dC = np.zeros(self.n.shape[0]) # chenge of concentration of A B C D E F
        k1 = A1 * np.exp( (-1 * E1) / (R * T))
        k2 = A2 * np.exp( (-1 * E2) / (R * T))
        k3 = A3 * np.exp( (-1 * E3) / (R * T))
        k4 = A4 * np.exp( (-1 * E4) / (R * T))
        k5 = A4 * np.exp( (-1 * E5) / (R * T))
        self.rate[0] = k1 * C[0] * C[1] * C[2] * dt # Rate of reaction A + B + C -> E
        self.rate[1] = k2 * C[0] * C[3] * dt # Rate of reaction A + D -> F
        self.rate[2] = k3 * C[1] * C[3] * dt # Rate of reaction B + D -> G
        self.rate[3] = k4 * C[2] * C[3] * dt # Rate of reaction C + D -> H
        self.rate[4] = k5 * C[5] * C[6] * C[7] * dt # Rate of reaction F + G + H -> |I|

        dC[0] = -1.0 * (self.rate[0] + self.rate[1])  # change of A
        dC[1] = -1.0 * (self.rate[0] + self.rate[2])  # change of B
        dC[2] = -1.0 * (self.rate[0] + self.rate[3]) # change of C
        dC[3] = -1.0 * (self.rate[1] + self.rate[2] + self.rate[3]) # change of D
        dC[4] = self.rate[0] # change of E
        dC[5] = self.rate[1] - self.rate[4] # change of F
        dC[6] = self.rate[2] - self.rate[4] # change of G
        dC[7] = self.rate[3] - self.rate[4] # change of H
        dC[8] = self.rate[4] # change of I
        # Update concentration of each chemical
        for i in range(self.n.shape[0]):
            self.n[i] += dC[i] * V * 1000 # need convert V from m^3 to L
        d_reward = dC[8] * V * 1000 # Reward is change in concentration of C
        return d_reward

    def get_total_pressure(self, V, T = 300):
        P_total = 0
        for i in range(self.n.shape[0]):
            P_total += self.n[i] * R * T / V
        return P_total

    def get_part_pressure(self, V, T = 300):
        P = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            P[i] = self.n[i] * R * T / V
        return P

    def get_concentration(self, V = 0.1):
        C = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            C[i] = self.n[i] / ( V * 1000)
        return C

    def get_spectra(self, V):
        # Initialize array for wavelength[0, 1] and absorbance
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        absorb = np.zeros(x.shape[0], dtype=np.float32)
        C = self.get_concentration(V)
        for i in range(len(self.params)): # for each species
            for j in range(self.params[i].shape[0]): # for each peak 
                for k in range(x.shape[0]):
                    absorb[k] += C[i] * self.params[i][j, 0] * np.exp(-0.5 * ((x[k] - self.params[i][j, 1]) / self.params[i][j, 2]) ** 2.0)
                                # amount    *    height      *  decay rate
        # Maximum possible absorbance at any wavelength is 1.0
        absorb = np.clip(absorb, 0.0, 1.0)
        return absorb

    def get_spectra_peak(self, V):
        spectra_peak = []
        C = self.get_concentration(V)
        spectra_peak.append([self.params[0][:, 1] * 600 + 200, C[0] * self.params[0][:, 0], 'A'])
        spectra_peak.append([self.params[1][:, 1] * 600 + 200, C[1] * self.params[1][:, 0], 'B'])
        spectra_peak.append([self.params[2][:, 1] * 600 + 200, C[2] * self.params[2][:, 0], 'C'])
        spectra_peak.append([self.params[3][:, 1] * 600 + 200, C[3] * self.params[3][:, 0], 'D'])
        spectra_peak.append([self.params[4][:, 1] * 600 + 200, C[4] * self.params[4][:, 0], 'E'])
        spectra_peak.append([self.params[5][:, 1] * 600 + 200, C[5] * self.params[5][:, 0], 'F'])
        spectra_peak.append([self.params[6][:, 1] * 600 + 200, C[6] * self.params[6][:, 0], 'G'])
        spectra_peak.append([self.params[7][:, 1] * 600 + 200, C[7] * self.params[7][:, 0], 'H'])
        spectra_peak.append([self.params[8][:, 1] * 600 + 200, C[8] * self.params[8][:, 0], 'I'])
        return spectra_peak

    def get_dash_line_spectra(self, V):
        dash_spectra = []
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        C = self.get_concentration(V)
        for i in range(len(self.params)): # for each species
            each_absorb = np.zeros(x.shape[0], dtype=np.float32)
            for j in range(self.params[i].shape[0]): # for each peak 
                for k in range(x.shape[0]):
                    each_absorb[k] += C[i] * self.params[i][j, 0] * np.exp(-0.5 * ((x[k] - self.params[i][j, 1]) / self.params[i][j, 2]) ** 2.0)
                                # amount    *    height      *  decay rate
            dash_spectra.append(each_absorb)
        return dash_spectra

    def plot_graph(self):
        A = 1.0 # Initial A
        B = 1.0 # Initial B
        C = 1.0 # Initial C
        D = 1.0 # Initial D
        E = 0.0 # Initial E
        F = 0.0 # Initial F
        G = 0.0 # Initial G
        H = 0.0 # Initial H
        I = 0.0 # Initial I

        T = 300.0 # Temperature (K)
        P = 101.325 # Pressure (kPa)
        dt = 0.01 # Time step (s)

        k1 = T / 300.0 # Rate of reaction 1
        k2 = T / 400.0 # Rate of reaction 2
        k3 = T / 200.0 # Rate of reaction 3
        k4 = T / 300.0 # Rate of reaction 4
        k5 = (T * 101.325) / (100.0 * P) # Rate of reaction 5

        n_steps = 500 # Number of steps to evolve system
        t = np.zeros(n_steps, dtype=np.float32) # Keep track of time for plotting
        conc = np.zeros((n_steps, 9), dtype=np.float32) # Keep track of concentrations for plotting
        conc[0, 0] = 1.0 * A
        conc[0, 1] = 1.0 * B
        conc[0, 2] = 1.0 * C
        conc[0, 3] = 1.0 * D
        conc[0, 4] = 1.0 * E
        conc[0, 5] = 1.0 * F
        conc[0, 6] = 1.0 * G
        conc[0, 7] = 1.0 * H
        conc[0, 8] = 1.0 * I

        for i in range(1, n_steps):
            # Calculate rate for this time step
            r1 = k1 * A * B * C
            r2 = k2 * A * D 
            r3 = k3 * B * D
            r4 = k4 * C * D
            r5 = k5 * F * G * H

            # Update concentrations
            A += -1.0 * (r1 + r2) * dt
            B += -1.0 * (r1 + r3) * dt
            C += -1.0 * (r1 + r4) * dt
            D += -1.0 * (r2 + r3 + r4) * dt
            E += r1 * dt
            F += (r2 - r5) * dt
            G += (r3 - r5) * dt
            H += (r4 - r5) * dt
            I += r5 * dt

            # Update plotting info
            conc[i, 0] = 1.0 * A
            conc[i, 1] = 1.0 * B
            conc[i, 2] = 1.0 * C
            conc[i, 3] = 1.0 * D
            conc[i, 4] = 1.0 * E
            conc[i, 5] = 1.0 * F
            conc[i, 6] = 1.0 * G
            conc[i, 7] = 1.0 * H
            conc[i, 8] = 1.0 * I
            t[i] = t[i-1] + dt

        plt.figure()
        for i in range(conc.shape[1]):
            plt.plot(t, conc[:, i], label=self.labels[i])
        plt.xlim([t[0], t[-1]])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (M)')
        plt.legend()
        plt.savefig('reaction_ex_3.pdf')
        plt.show()
        plt.close()
