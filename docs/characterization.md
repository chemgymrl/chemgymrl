[chemgymrl.com](https://chemgymrl.com/)

## Characterization Bench

<span style="display:block;text-align:center">![Characterization](tutorial_figures/characterization.png)

The characterization bench is the primary method in which an agent or lab manager can look inside vessel containers. The purpose of the characterization bench is not to manipulate the inputted vessel, but subject it to analysis techniques that observe the state of the vessel including the materials inside it and their relative quantities. This does not mean that the contents of the inputted vessel cannot be modified by the characterization bench. This allows an agent or lab manager to observe vessels, determine their contents, and allocate the vessel to the necessary bench for further experimentation.
 

Included with the characterization bench are several analysis techniques available to the agent working at this bench. The foremost of which is spectrometric analysis. Such analysis includes the observation of spectra being emitted by materials. Performing spectrometry techniques on a vessel produces spectral signatures giving insight into the contents of the vessel. Different materials possess different spectral signatures and cross-checking the produced spectral data with the known spectral signatures of supported materials allows the agent to identify the vessel’s contents. Additional characterization bench techniques are in development.
 

In observing the state of the vessel, the agent or lab manager operating this bench can allocate the vessel to another bench for further experimentation in the pursuit of completing the required task. For example, in the extraction of salt from water, if a vessel is determined to have much more water than salt, the lab manager can use this information to perform further extractions, increasing the purity of salt in the vessel, or subject the vessel to the reaction bench and increase the amount of salt present in the vessel.

## Input

The input to the characterization bench is a vessel, and a desired analysis technique to perform on the vessel. These are 
passed in as parameters the analyze method is called.

```python
def analyze(self, vessel, analysis, overlap=False):
        """
        Constructor class method to pass thermodynamic variables to class methods.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # perform the specified analysis technique
        analysis = self.techniques[analysis](vessel, overlap)
        return analysis, vessel
```

## Output

The output to the characterization bench will depend on the analysis technique called. For example, if the 
characterization bench is called to perform an absorption spectra analysis, it will return a spectral graph back to the 
agent. This output is an array filled spectral signatures.

```python
array([5.00000007e-02, 3.88000011e-01, 5.02251148e-01, 1.09714296e-04,
       4.81336862e-01, 4.81336862e-01, 4.81336862e-01, 4.44010586e-01,
       5.13693636e-19, 7.02555729e-17, 7.03501986e-15, 5.15772321e-13,
       2.76858796e-11, 1.08809528e-09, 3.13100337e-08, 6.59643604e-07,
       1.01759824e-05, 1.15057468e-04, 9.62913211e-04, 6.32128678e-03,
       3.75580229e-02, 2.00341001e-01, 7.17306733e-01, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00, 3.45313042e-01, 1.44143984e-01,
       5.09813428e-02, 1.36140008e-02, 2.67074024e-03, 3.83689068e-04,
       4.03588565e-05, 3.10818086e-06, 1.75260325e-07, 7.23544025e-09,
       2.18704929e-10, 4.84011182e-12, 7.84269364e-14, 9.30416882e-16,
       8.08323459e-18, 1.12275603e-19, 2.05226074e-18, 5.80436606e-17,
       1.37783885e-15, 2.74536897e-14, 4.59490844e-13, 6.47546600e-12,
       7.74089057e-11, 8.00548128e-10, 7.47670370e-09, 6.72209808e-08,
       6.05021455e-07, 5.25721134e-06, 4.06777654e-05, 2.62283138e-04,
       1.35838834e-03, 5.55886794e-03, 1.79146975e-02, 4.58729863e-02,
       9.56562534e-02, 1.71522617e-01, 4.47697878e-01, 1.00000000e+00,
       7.51069903e-01, 9.40195024e-01, 9.00364399e-01, 6.52727604e-01,
       4.12488669e-01, 4.53188211e-01, 7.02614784e-01, 6.33801460e-01,
       5.42857647e-01, 7.87446022e-01, 8.28699231e-01, 5.09403706e-01,
       3.42276931e-01, 4.80235159e-01, 5.98179400e-01, 4.64588821e-01,
       3.06268275e-01, 6.96131766e-01, 2.84563184e-01, 3.83535400e-02,
       1.35065034e-01, 3.58249873e-01, 5.67879021e-01, 5.37672698e-01,
       3.04067552e-01, 1.02939621e-01, 3.68590318e-02, 2.35721827e-01,
       6.96323574e-01, 4.51030999e-01, 2.86931723e-01, 6.88235164e-01,
       6.71657443e-01, 5.08716106e-01, 7.10071266e-01, 7.77660191e-01,
       5.14985025e-01, 2.03744531e-01, 4.81469259e-02, 6.80406019e-03,
       8.61246954e-04, 5.11169108e-03, 4.54575084e-02, 2.06235260e-01,
       4.74683940e-01, 5.54266274e-01, 3.28337371e-01, 9.92386937e-02,
       2.62317788e-02, 8.07170495e-02, 2.06026495e-01, 1.94233477e-01,
       6.67030066e-02, 8.34272243e-03, 5.45541698e-04, 3.24273586e-01,
       1.61323464e-03, 7.61622823e-06, 1.79047114e-04, 2.51422916e-03,
       2.10876912e-02, 1.05641104e-01, 3.16106349e-01, 5.64964831e-01,
       6.03126884e-01, 3.84699106e-01, 1.47269651e-01, 3.68326940e-02,
       1.63703002e-02, 3.14945281e-02, 6.49587736e-02, 1.14196159e-01,
       1.93104148e-01, 3.44255805e-01, 5.76812506e-01, 7.60537386e-01,
       7.16183782e-01, 4.64241356e-01, 2.04477564e-01, 6.09111637e-02,
       1.22490944e-02, 1.66183745e-03, 1.53511064e-04, 3.13458440e-05,
       2.25370997e-04, 1.55353406e-03, 7.23012397e-03, 2.26783678e-02,
       4.79432717e-02, 6.83103353e-02, 6.55978024e-02, 4.24556844e-02,
       1.85195580e-02, 5.44459885e-03, 1.08004955e-03, 2.12187893e-04,
       1.85946433e-03, 2.48207953e-02, 1.65429294e-01, 5.46763480e-01,
       8.96088779e-01, 7.28235900e-01, 2.93466330e-01, 5.86434752e-02,
       5.81085170e-03, 2.85512448e-04, 6.95658900e-06, 8.40456948e-08,
       5.03532993e-10, 1.49582571e-12, 2.20342958e-15, 1.60961313e-18,
       5.83006781e-22, 1.04721791e-25, 9.32659336e-30, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
       , dtype=float32)
```

Note the figure above only shows a snippet of the values returned, in reality the array is much larger.