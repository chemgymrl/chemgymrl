'''
The material construct signature:

Property:
    - `smiles` (str): The smiles code of the material
    - `name` (str): The name of the material.
    - `molar_mass` (float): The average molar mass of the material.
    - `phase` (str): The state of the material at the room temperature.
    - `dissolves_in` (set): A set of smiles codes for materials this dissolves in
    - `polarity` (float): The polarity of the material
    - `dissolved_components` (dict): A dictionary of (smiles_code,mol) pairs of what / how much material(s) are dissociated when one mol of this material is dissolved.
    - `boiling_point_K` (float): The material's boiling point in Kelvin.
    - `melting_point_K` (float): The material's melting point in Kelvin.
    - `specific_heat_J_gK` (float): The material's specific heat capacity in Joules per gram*Kelvin
    - `enthalpy_fusion_J_mol` (float): The material's fusion enthalpy in Joules per mol
    - `enthalpy_vapor_J_mol` (float): The material's vaporization enthalpy in Joules per mol
    - `density_g_ml` (dict): The material's density in each phase in grams per mililitre
    - `color` (float): Single channel color for the material (will be depricated)
    - `n` (float): The material's refractive index
    - `spectra_overlap` (np.array): Spectral information of the material
    - `spectra_no_overlap` (np.array): Alternative? spectral information of the material
'''


import pandas as pd
import numpy as np
import os

REGISTRY = dict()

construct_signature = ('smiles','name','molar_mass','phase','dissolves_in','polarity','dissolved_components',
        'boiling_point_K','melting_point_K','specific_heat_J_gK','enthalpy_fusion_J_mol','enthalpy_vapor_J_mol',
        'density_g_ml','color','n','spectra_overlap','spectra_no_overlap')

def register(*material_kwargs):
    """
    Add a material or list of materials to the materials registry.

    The materials should either be represented via a tuple corresponding to construct_signature, or a dictionary
    with construct_signature's items as keys.
    
    """
    for kwargs in material_kwargs:
        if not (type(kwargs) is dict):
            kwargs = {key:val for key,val in zip(construct_signature,kwargs)}
        key = kwargs["smiles"]
        if key in REGISTRY:
            raise Exception(f"Cannot register the same Material ({key}) Twice!")
        REGISTRY[key] = MaterialConstructor(kwargs)

def from_smiles(smiles_string, mol=0):
    """
    Args:
        smiles_string (str): The smiles code corresponding to the material.
        mol (float): The amount of material in moles.

    Returns:
        Material: An instance of the material specified by smiles_string, with it's amount set to mol.
    """
    return REGISTRY[smiles_string](mol=mol)

class MaterialConstructor():
    def __init__(self, kwargs: dict):
        assert not "mol" in kwargs
        Material(**kwargs)
        self.kwargs=kwargs
    def get_args(self):
        return tuple(self.kwargs.get(key) for key in construct_signature)
    def __call__(self,mol=0):
        return Material(mol=mol,**self.kwargs)
    def __repr__(self):
        return "Material"+self.args_repr()
    def args_repr(self,pad=0):
        """
        String format the material info so it can be easily conostructed
        """
        def string_cast(s,i=[0]):
            #handle numpy arrays differently
            if type(s) is np.ndarray:
                np.set_printoptions(threshold=np.inf)
                out = "np."+np.array_repr(s, max_line_width=np.inf).replace("\n", "").replace("       ", "")
                if s.size==0:
                    out = out.replace("[], ","").replace("np.","np.nd")
            else:
                out = repr(s)
            #don't think about this too hard
            if type(pad) is int:
                p=pad
            else:
                p=pad[i[0]]
                i[0]=i[0]+1
            return out.ljust(p) if p>0 else out
        args = tuple(self.kwargs.get(key) for key in construct_signature)
        return ("("+",".join(string_cast(a) for a in args)+")")

class Material():
    def __init__(
        self,
        smiles,
        name,
        molar_mass,
        phase = 'l',
        dissolves_in = set(),
        polarity=0,
        dissolved_components=dict(),
        boiling_point_K = 1e10,  # in K
        melting_point_K = 0,  # in K
        specific_heat_J_gK = 1.0,  # in J/g*K
        enthalpy_fusion_J_mol=1.0,  # in J/mol
        enthalpy_vapor_J_mol=1.0,  # in J/mol
        density_g_ml = dict(s=1,l=1,g=1),
        color=0,
        n=1,
        spectra_overlap=None,
        spectra_no_overlap=None,
        mol = 0,
    ):

        self.phase=phase
        self.polarity = polarity
        self.color=color
        self.mol=mol

        self._solute=False
        self._solvent=False

        #These should never change (different materials may point to the same value)
        self._smiles=smiles
        self._name=name
        self._molar_mass_g_mol = molar_mass
        self._dissolves_in = dissolves_in
        self._dissolved_components = dissolved_components
        self._boiling_point_K=boiling_point_K
        self._melting_point_K=melting_point_K
        self._specific_heat_J_gK=specific_heat_J_gK
        self._enthalpy_fusion_J_mol=enthalpy_fusion_J_mol
        self._enthalpy_vapor_J_mol=enthalpy_vapor_J_mol
        self._density_g_ml=density_g_ml
        self._n = n
        self._spectra_overlap=spectra_overlap
        self._spectra_no_overlap=spectra_no_overlap
    
    def __repr__(self):
        return self._smiles
    def __hash__(self):
        return hash(self._smiles)
    def __eq__(self,other):
        return self._smiles==other._smiles
    @property
    def molar_mass(self):
        return self._molar_mass_g_mol
    @property
    def boiling_point_K(self):
        return self._boiling_point_K
    @property
    def melting_point_K(self):
        return self._melting_point_K
    #Derived quantities
    @property
    def heat_capacity_J_K(self):
        return self.mol*self._molar_mass_g_mol*self._specific_heat_J_gK
    @property
    def volume_L(self):
        return 1e-3*self.mol*self._molar_mass_g_mol/self._density_g_ml[self.phase]
    @property
    def litres_per_mol(self):
        return 1e-3*self._molar_mass_g_mol/self._density_g_ml[self.phase]
    @property
    def vapour_enthalpy_J(self):
        return self.mol*self._enthalpy_vapor_J_mol
    @property
    def density_g_L(self):
        return self._density_g_ml[self.phase] * 1000
    @property
    def is_solute(self):
        return self._solute
    @is_solute.setter
    def is_solute(self, flag):
        self._solute = bool(flag)
        self._solvent = self._solvent and not self._solute

    @property
    def is_solvent(self):
        return self._solvent

    @property
    def transmittance(self):
        return (1-((1-self._n)/(1+self._n))**2)**2

    @is_solvent.setter
    def is_solvent(self,flag):
        self._solvent = bool(flag)
        self._solute = self._solute and not self._solvent

        
    def ration(self,ratio):
        """
        Creates a new Material and moves `ratio` fraction of the mols to the new material
        Args:
            ratio (float): Should be in [0,1]
        Returns
            Material: a copy of the material with the moved mols.
        """
        diff=ratio*self.mol
        self.mol -= diff
        mat = REGISTRY[self._smiles](diff)
        mat.phase=self.phase
        mat.polarity=self.polarity
        # might remove later. . .
        mat._solvent=self._solvent
        mat._solute=self._solute
        return mat

    def all_dissolves_in(self):
        """
        Returns:
            Generator[str]: A generator of smiles codes representing the Materials this material dissolves in."""
        return (key for key in self._dissolves_in)

    def get_spectra_overlap(self):
        return self._spectra_overlap

    def get_spectra_no_overlap(self):
        return self._spectra_no_overlap

    def dissolve(self):
        return self._dissolved_components



from numpy import float32,float64
nan = np.nan
_MATERIALS = (
# smiles                                  name                     molar_mass phase dissolves_in                   polarity              dissolved_components                      boiling_point_K  melting_point_K specific_heat_J_gK enthalpy_fusion_J_mol enthalpy_vapor_J_mol density_g_ml                            color   n           spectra_overlap                          spectra_no_overlap
('Air'                                   ,'Air'                    ,28.963    ,'g'  ,set()                         ,0.0                 ,{'Air': 1}                                ,1.0            ,1.0             ,1.0035            ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 0.001225}   ,0.65, 1.003       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('O'                                     ,'H2O'                    ,18.015    ,'l'  ,set()                         ,1.4313200712923395  ,{'O': 1}                                  ,373.15         ,1.0             ,4.1813            ,1.0                 ,40650.0             ,{'s': None, 'l': 0.997, 'g': None}      ,0.2 , 1.333       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[H]'                                   ,'H'                      ,1.008     ,'g'  ,set()                         ,0.0                 ,{'[H]': 1}                                ,20.25          ,1.0             ,1.0               ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 8.9e-05}    ,0.1 , 1           ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[HH]'                                  ,'H2'                     ,2.016     ,'g'  ,set()                         ,0.0                 ,{'[HH]': 1}                               ,20.25          ,1.0             ,14.304            ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 8.9e-05}    ,0.1 , 1.000149    ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[O]'                                   ,'O'                      ,15.999    ,'g'  ,set()                         ,0.0                 ,{'[O]': 1}                                ,90.188         ,1.0             ,nan               ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 0.001429}   ,0.15, 1           ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('O=O'                                   ,'O2'                     ,31.999    ,'g'  ,set()                         ,0.0                 ,{'O=O': 1}                                ,90.188         ,1.0             ,0.918             ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 0.001429}   ,0.1 , 1.000291    ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[O-][O+]=O'                            ,'O3'                     ,47.998    ,'g'  ,set()                         ,0.047971811940158204,{'[O-][O+]=O': 1}                         ,161.15         ,1.0             ,nan               ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 0.002144}   ,0.1 , 1.278       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCCCC'                                ,'C6H14'                  ,86.175    ,'l'  ,set()                         ,0.0                 ,{'CCCCCC': 1}                             ,342.15         ,1.0             ,2.26              ,1.0                 ,1.0                 ,{'s': None, 'l': 0.655, 'g': None}      ,0.9 , 1.3758      ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Na+].[Cl-]'                           ,'NaCl'                   ,58.443    ,'s'  ,set()                         ,1.5                 ,{'[Na+]': 1, '[Cl-]': 1}                  ,1738.0         ,1.0             ,0.853             ,27950.0             ,229700.0            ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.4 , 1.5442      ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Na+]'                                 ,'Na'                     ,22.99     ,'s'  ,{'CCOCC', 'CCCCCC', 'O'}      ,2.0                 ,{'[Na+]': 1}                              ,1156.0         ,1.0             ,1.23              ,2600.0              ,97700.0             ,{'s': 0.968, 'l': 0.856, 'g': None}     ,0.85, 0.049195    ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Cl-]'                                 ,'Cl'                     ,35.453    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,2.0                 ,{'[Cl-]': 1}                              ,1156.0         ,1.0             ,0.48              ,3200.0              ,10200.0             ,{'s': None, 'l': 1.558, 'g': 0.003214}  ,0.8 , 1           ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CLCL'                                  ,'Cl2'                    ,70.906    ,'g'  ,set()                         ,0.0                 ,{'CLCL': 1}                               ,238.55         ,1.0             ,1.0               ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 0.002898}   ,0.8 , 1.375       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Li+].[F-]'                            ,'LiF'                    ,25.939    ,'s'  ,{'O'}                         ,1.5                 ,{'[Li+].[F-]': 1}                         ,1953.15        ,1.0             ,1.0               ,1.0                 ,1.0                 ,{'s': 2.64, 'l': None, 'g': None}       ,0.9 , 1.4069      ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Li+]'                                 ,'Li'                     ,6.941     ,'s'  ,{'O'}                         ,0.0                 ,{'[Li+]': 1}                              ,1603.15        ,1.0             ,1.0               ,1.0                 ,1.0                 ,{'s': 0.534, 'l': None, 'g': None}      ,0.95, 0.34301     ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('FF'                                    ,'F2'                     ,37.997    ,'g'  ,set()                         ,0.0                 ,{'FF': 1}                                 ,85.15          ,1.0             ,0.824             ,1.0                 ,1.0                 ,{'s': None, 'l': None, 'g': 0.001696}   ,0.8 , 1.092       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[O-]S(=O)(=O)[O-].[Cu+2]'              ,'CuS04'                  ,159.6     ,'s'  ,{'O'}                         ,1.5                 ,{'[O-]S(=O)(=O)[O-].[Cu+2]': 1}           ,923.0          ,383.0           ,0.853             ,27950.0             ,229700.0            ,{'s': 3.6, 'l': None, 'g': None}        ,0.9 , 1           ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O'    ,'CuS04*5H2O'             ,249.68    ,'s'  ,set()                         ,1.5                 ,{'[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O': 1} ,923.0          ,383.0           ,0.853             ,27950.0             ,229700.0            ,{'s': 2.286, 'l': None, 'g': None}      ,0.9 , 1           ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCCCCCCCCCC'                          ,'dodecane'               ,170.34    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,0.0                 ,{'CCCCCCCCCCCC': 1}                       ,489.5          ,263.6           ,2.3889            ,19790.0             ,41530.0             ,{'s': None, 'l': 0.75, 'g': None}       ,0.15, 1.422       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCCCCCl'                              ,'1-chlorohexane'         ,120.62    ,'l'  ,{'CCCCCC'}                    ,0.0                 ,{'CCCCCCCl': 1}                           ,408.2          ,179.2           ,1.5408            ,15490.0             ,42800.0             ,{'s': None, 'l': 0.879, 'g': None}      ,0.1 , 1.419       ,np.array([[7.50e-01, 2.95e+03, 1.00e-03],[8.32e-01, 2.94e+03, 2.00e-03],[3.88e-01, 2.88e+03, 9.00e-03],[1.33e-01, 1.45e+03, 1.00e-02],[9.00e-02, 1.30e+03, 1.50e-02],[1.08e-01, 7.35e+02, 1.00e-02],[6.20e-02, 6.60e+02, 8.00e-03]]),np.array([[7.50e-01, 2.95e+03, 1.00e-03],[8.32e-01, 2.94e+03, 2.00e-03],[3.88e-01, 2.88e+03, 9.00e-03],[1.33e-01, 1.45e+03, 1.00e-02],[9.00e-02, 1.30e+03, 1.50e-02],[1.08e-01, 7.35e+02, 1.00e-02],[6.20e-02, 6.60e+02, 8.00e-03]])),
('CCCCC(C)Cl'                            ,'2-chlorohexane'         ,120.62    ,'l'  ,{'CCCCCC'}                    ,0.0                 ,{'CCCCC(C)Cl': 1}                         ,395.2          ,308.3           ,1.5408            ,11970.0             ,43820.0             ,{'s': None, 'l': 0.87, 'g': None}       ,0.15, 1.412       ,np.array([[9.30e-01, 2.95e+03, 5.60e-03],[9.00e-01, 2.85e+03, 1.70e-03],[7.70e-01, 1.44e+03, 1.80e-03],[7.90e-01, 1.39e+03, 8.80e-03],[5.20e-01, 1.31e+03, 5.00e-03],[7.00e-01, 1.26e+03, 6.00e-03],[5.20e-01, 1.20e+03, 7.00e-03],[5.90e-01, 1.16e+03, 3.00e-03],[5.20e-01, 1.10e+03, 7.00e-03],[6.30e-01, 1.04e+03, 4.00e-03],[6.30e-01, 1.01e+03, 4.50e-03],[7.00e-01, 9.80e+02, 7.00e-03],[5.00e-01, 9.00e+02, 6.10e-03],[2.00e-01, 8.60e+02, 5.00e-03],[3.00e-01, 8.30e+02, 1.40e-03],[5.50e-01, 7.80e+02, 7.00e-03],[6.50e-01, 7.20e+02, 8.00e-03],[8.00e-01, 6.20e+02, 6.00e-03]]),np.array([[9.30e-01, 2.95e+03, 5.60e-03],[9.00e-01, 2.85e+03, 1.70e-03],[7.70e-01, 1.44e+03, 1.80e-03],[7.90e-01, 1.39e+03, 8.80e-03],[5.20e-01, 1.31e+03, 5.00e-03],[7.00e-01, 1.26e+03, 6.00e-03],[5.20e-01, 1.20e+03, 7.00e-03],[5.90e-01, 1.16e+03, 3.00e-03],[5.20e-01, 1.10e+03, 7.00e-03],[6.30e-01, 1.04e+03, 4.00e-03],[6.30e-01, 1.01e+03, 4.50e-03],[7.00e-01, 9.80e+02, 7.00e-03],[5.00e-01, 9.00e+02, 6.10e-03],[2.00e-01, 8.60e+02, 5.00e-03],[3.00e-01, 8.30e+02, 1.40e-03],[5.50e-01, 7.80e+02, 7.00e-03],[6.50e-01, 7.20e+02, 8.00e-03],[8.00e-01, 6.20e+02, 6.00e-03]])),
('CCCC(CC)Cl'                            ,'3-chlorohexane'         ,120.62    ,'l'  ,{'CCCCCC'}                    ,0.0                 ,{'CCCC(CC)Cl': 1}                         ,396.2          ,308.3           ,1.5408            ,11970.0             ,32950.0             ,{'s': None, 'l': 0.9, 'g': None}        ,0.2 , 1.412       ,np.array([[2.930e-02, 2.966e+03, 6.000e-03],[1.180e-02, 2.889e+03, 2.000e-03],[4.800e-03, 1.462e+03, 6.000e-03]]),np.array([[2.930e-02, 2.966e+03, 6.000e-03],[1.180e-02, 2.889e+03, 2.000e-03],[4.800e-03, 1.462e+03, 6.000e-03]])),
('CCCCCCC(C)CCCC'                        ,'5-methylundecane'       ,170.34    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,0.0                 ,{'CCCCCCC(C)CCCC': 1}                     ,481.1          ,255.2           ,2.3889            ,19790.0             ,41530.0             ,{'s': None, 'l': 0.75, 'g': None}       ,0.25, 1.421       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCCCCC(CC)CCC'                        ,'4-ethyldecane'          ,170.34    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,0.0                 ,{'CCCCCCC(CC)CCC': 1}                     ,480.1          ,254.2           ,2.3889            ,19790.0             ,41530.0             ,{'s': None, 'l': 0.75, 'g': None}       ,0.3 , 1.421       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCCC(C)C(C)CCCC'                      ,'5,6-dimethyldecane'     ,170.34    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,0.0                 ,{'CCCCC(C)C(C)CCCC': 1}                   ,474.2          ,222.4           ,2.3889            ,19790.0             ,41530.0             ,{'s': None, 'l': 0.757, 'g': None}      ,0.35, 1.42        ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCCC(C)C(CC)CCC'                      ,'4-ethyl-5-methylnonane' ,170.34    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,0.0                 ,{'CCCCC(C)C(CC)CCC': 1}                   ,476.3          ,224.5           ,2.3889            ,19790.0             ,41530.0             ,{'s': None, 'l': 0.75, 'g': None}       ,0.4 , 1.42        ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCCC(CC)C(CC)CCC'                      ,'4,5-diethyloctane'      ,170.34    ,'l'  ,{'CCOCC', 'CCCCCC', 'O'}      ,0.0                 ,{'CCCC(CC)C(CC)CCC': 1}                   ,470.2          ,222.4           ,2.3889            ,19790.0             ,41530.0             ,{'s': None, 'l': 0.768, 'g': None}      ,0.45, 1.42        ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCOC(=O)C'                             ,'ethyl acetate'          ,88.106    ,'l'  ,{'O'}                         ,0.654               ,{'CCOC(=O)C': 1}                          ,350.0          ,189.6           ,1.904             ,10480.0             ,31940.0             ,{'s': None, 'l': 0.902, 'g': None}      ,0.4 , 1.372       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('CCOCC'                                 ,'diethyl ether'          ,74.123    ,'l'  ,set()                         ,1.3                 ,{'CCOCC': 1}                              ,307.8          ,156.8           ,119.46            ,-252700.0           ,27247.0             ,{'s': None, 'l': 0.7134, 'g': None}     ,0.05, 1.353       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('[Af]'                                  ,'fict_A'                 ,170.34    ,'l'  ,{'O'}                         ,0.0                 ,{'[Af]': 1}                               ,489.5          ,263.6           ,2.3889            ,19790.0             ,41530.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.15, 1           ,np.array([[9.43904295e-01, 2.41678874e+03, 9.95871813e-03],[9.93508858e-01, 1.61688257e+03, 1.95518549e-02],[5.04060614e-01, 4.12622244e+02, 2.64196495e-03],[1.57465892e-01, 7.07813158e+02, 7.55995404e-04]]),np.array([[9.43904295e-01, 2.41678874e+03, 9.95871813e-03],[9.93508858e-01, 1.61688257e+03, 1.95518549e-02],[5.04060614e-01, 4.12622244e+02, 2.64196495e-03],[1.57465892e-01, 7.07813158e+02, 7.55995404e-04]])),
('[Bf]'                                  ,'fict_B'                 ,120.62    ,'l'  ,{'O'}                         ,0.0                 ,{'[Bf]': 1}                               ,408.2          ,179.2           ,1.5408            ,15490.0             ,42800.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.1 , 1           ,np.array([[1.85140447e-01, 2.87863429e+03, 7.13364826e-03],[9.18588276e-01, 7.58568779e+02, 7.38481312e-03],[7.38631244e-02, 1.55234830e+03, 3.13263193e-02]]),np.array([[1.85140447e-01, 2.87863429e+03, 7.13364826e-03],[9.18588276e-01, 7.58568779e+02, 7.38481312e-03],[7.38631244e-02, 1.55234830e+03, 3.13263193e-02]])),
('[Cf]'                                  ,'fict_C'                 ,120.62    ,'l'  ,{'O'}                         ,0.0                 ,{'[Cf]': 1}                               ,395.2          ,308.3           ,1.5408            ,11970.0             ,43820.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.15, 1           ,np.array([[5.28979037e-01, 2.99592773e+03, 3.29373665e-02],[9.43872169e-01, 2.37783762e+03, 6.15995397e-03],[6.13353671e-01, 2.98201571e+03, 9.85076476e-03]]),np.array([[5.28979037e-01, 2.99592773e+03, 3.29373665e-02],[9.43872169e-01, 2.37783762e+03, 6.15995397e-03],[6.13353671e-01, 2.98201571e+03, 9.85076476e-03]])),
('[Df]'                                  ,'fict_D'                 ,120.62    ,'l'  ,{'O'}                         ,0.0                 ,{'[Df]': 1}                               ,396.2          ,308.3           ,1.5408            ,11970.0             ,32950.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.2 , 1           ,np.array([[1.90579725e-01, 2.55963467e+03, 3.49184581e-03]]),np.array([[1.90579725e-01, 2.55963467e+03, 3.49184581e-03]])),
('[Ef]'                                  ,'fict_E'                 ,170.34    ,'l'  ,{'O'}                         ,0.0                 ,{'[Ef]': 1}                               ,481.1          ,255.2           ,2.3889            ,19790.0             ,41530.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.25, 1           ,np.array([[1.81834079e-01, 1.26477724e+03, 1.36971571e-03],[6.54988735e-02, 7.62889243e+02, 1.89455691e-03]]),np.array([[1.81834079e-01, 1.26477724e+03, 1.36971571e-03],[6.54988735e-02, 7.62889243e+02, 1.89455691e-03]])),
('[Ff]'                                  ,'fict_F'                 ,170.34    ,'l'  ,{'O'}                         ,0.0                 ,{'[Ff]': 1}                               ,480.1          ,254.2           ,2.3889            ,19790.0             ,41530.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.3 , 1           ,np.array([[4.20758409e-01, 2.46074517e+02, 5.57671503e-02]]),np.array([[4.20758409e-01, 2.46074517e+02, 5.57671503e-02]])),
('[Gf]'                                  ,'fict_G'                 ,170.34    ,'l'  ,{'O'}                         ,0.0                 ,{'[Gf]': 1}                               ,474.2          ,222.4           ,2.3889            ,19790.0             ,41530.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.35, 1           ,np.array([[5.03802045e-01, 2.21412634e+03, 1.33214062e-03],[9.30694466e-01, 5.43742572e+02, 9.58193826e-03],[8.58834516e-01, 1.46912516e+03, 3.04494718e-03]]),np.array([[5.03802045e-01, 2.21412634e+03, 1.33214062e-03],[9.30694466e-01, 5.43742572e+02, 9.58193826e-03],[8.58834516e-01, 1.46912516e+03, 3.04494718e-03]])),
('[Hf]'                                  ,'fict_H'                 ,170.34    ,'l'  ,{'O'}                         ,0.0                 ,{'[Hf]': 1}                               ,476.3          ,224.5           ,2.3889            ,19790.0             ,41530.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.4 , 1           ,np.array([[9.77456880e-01, 2.61364890e+03, 8.92133157e-03],[9.60887633e-01, 2.95398827e+03, 8.55542947e-02]]),np.array([[9.77456880e-01, 2.61364890e+03, 8.92133157e-03],[9.60887633e-01, 2.95398827e+03, 8.55542947e-02]])),
('[If]'                                  ,'fict_I'                 ,170.34    ,'l'  ,{'O'}                         ,0.0                 ,{'[If]': 1}                               ,470.2          ,222.4           ,2.3889            ,19790.0             ,41530.0             ,{'s': 2.165, 'l': 2.165, 'g': None}     ,0.45, 1           ,np.array([[1.03611024e-01, 1.06505255e+03, 1.63771049e-03]]),np.array([[1.03611024e-01, 1.06505255e+03, 1.63771049e-03]])),
('CN(C)c1ccc(cc1)/N=N/c2ccccc2C(=O)O'    ,'methyl red'             ,88.106    ,'s'  ,set()                         ,0.0                 ,{'CN(C)c1ccc(cc1)/N=N/c2ccccc2C(=O)O': 1} ,630.0          ,455.0           ,1.904             ,10480.0             ,31940.0             ,{'s': 0.902, 'l': None, 'g': None}      ,0.6 , 1.593       ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
('Cl'                                    ,'HCl'                    ,88.106    ,'g'  ,{'CCOCC', 'O'}                ,0.0                 ,{'Cl': 1}                                 ,350.0          ,189.6           ,1.904             ,10480.0             ,31940.0             ,{'s': None, 'l': None, 'g': 0.00148}    ,0.3 , 1           ,np.ndarray(shape=(0, 3), dtype=float64) ,np.ndarray(shape=(0, 3), dtype=float64) ),
)


register(*_MATERIALS)


