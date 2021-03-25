import numpy as np

class AnalysisBench:
    def __init__(self):
        self.techniques = {'spectra': self.get_spectra}

    def analyze(self, vessel, analysis, overlap=False):
        analysis = self.techniques[analysis](vessel, overlap)
        return analysis

    def get_spectra(self, vessel, overlap):
        '''
        Class method to generate total spectral data using a guassian decay.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in Litres

        Returns
        ---------------
        absorb : np.array
            An array of the total absorption data of every chemical in the experiment

        Raises
        ---------------
        None
        '''
        if not overlap:
            params = [item[0]().get_spectra_no_overlap() for __, item in vessel.get_material_dict().items()]
        else:
            params = [item[0]().get_spectra_overlap() for __, item in vessel.get_material_dict().items()]

        C = vessel.get_concentration()
        # set the wavelength space
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        # define an array to contain absorption data
        absorb = np.zeros(x.shape[0], dtype=np.float32)

        # obtain the concentration array
        # C = self.get_concentration(V)

        # iterate through the spectral parameters in self.params and the wavelength space
        for i, item in enumerate(params):
            for j in range(item.shape[0]):
                for k in range(x.shape[0]):
                    amount = C[i]
                    height = item[j, 0]
                    decay_rate = np.exp(
                        -0.5 * (
                            (x[k] - params[i][j, 1]) / params[i][j, 2]
                        ) ** 2.0
                    )
                    if decay_rate < 1e-30:
                        decay_rate = 0
                    absorb[k] += amount * height * decay_rate

        # absorption must be between 0 and 1
        absorb = np.clip(absorb, 0.0, 1.0)

        return absorb

    def get_spectra_peak(self, V, C, params, materials):
        '''
        Method to populate a list with the spectral peak of each chemical.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in litres.

        Returns
        ---------------
        spectra_peak : list
            A list of parameters specifying the peak of the spectra for each chemical

        Raises
        ---------------
        None
        '''


        # create a list of the spectral peak of each chemical
        spectra_peak = []
        for i, material in enumerate(materials):
            spectra_peak.append([
                params[i][:, 1] * 600 + 200,
                C[i] * params[i][:, 0],
                material
            ])
        return spectra_peak

    def get_dash_line_spectra(self, V, C, params):
        '''
        Module to generate each individual spectral dataset using gaussian decay.

        Parameters
        ---------------
        V : np.float32
            The volume of the system in Litres

        Returns
        ---------------
        dash_spectra : list
            A list of all the spectral data of each chemical

        Raises
        ---------------
        None
        '''

        dash_spectra = []

        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)

        for i, item in enumerate(params):
            each_absorb = np.zeros(x.shape[0], dtype=np.float32)
            for j in range(item.shape[0]):
                for k in range(x.shape[0]):
                    amount = C[i]
                    height = item[j, 0]
                    decay_rate = np.exp(
                        -0.5 * (
                            (x[k] - params[i][j, 1]) / params[i][j, 2]
                        ) ** 2.0
                    )
                    each_absorb += amount * height * decay_rate
            dash_spectra.append(each_absorb)

        return dash_spectra
