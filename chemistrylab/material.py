import pandas as pd
import os

REGISTRY = dict()

construct_signature = ('smiles','name','molar_mass','phase','dissolves_in','polarity','dissolved_components',
        'boiling_point_K','melting_point_K','specific_heat_J_gK','enthalpy_fusion_J_mol','enthalpy_vapor_J_mol',
        'density_g_ml','spectra_overlap','spectra_no_overlap','color')

def register(*material_kwargs):
    for kwargs in material_kwargs:
        if not (type(kwargs) is dict):
            kwargs = {key:val for key,val in zip(construct_signature,kwargs)}
        key = kwargs["smiles"]
        if key in REGISTRY:
            raise Exception(f"Cannot register the same Material ({key}) Twice!")
        REGISTRY[key] = MaterialConstructor(kwargs)

class MaterialConstructor():
    def __init__(self, kwargs: dict):
        assert not "mol" in kwargs
        Material(**kwargs)
        self.kwargs=kwargs
    def get_args(self):
        return tuple(self.kwargs.get(key) for key in construct_signature)
    def __call__(self,mol=0):
        return Material(mol=mol,**self.kwargs)

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
        spectra_overlap=None,
        spectra_no_overlap=None,
        color=0,
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
    @is_solvent.setter
    def is_solvent(self,flag):
        self._solvent = bool(flag)
        self._solute = self._solute and not self._solvent

        
    def ration(self,ratio):
        """
        Creates a new Material and moves `ratio` fraction of the moles to the new material
        Args:
        - ratio (float): Should be in [0,1]
        Returns
        - mat (Material): Material with the same class where the moles were moved
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

    def all_dissolved_in(self):
        """
        Returns:
        - Generator[str]: A generator of smiles codes representing the Materials this material dissolves in."""
        return (key for key in self._dissolves_in)

    def get_spectra_overlap(self):
        return self._spectra_overlap

    def get_spectra_no_overlap(self):
        return self._spectra_no_overlap

    def dissolve(self):
        return self._dissolved_components


def from_old_mat(mat):
    kwargs = dict(
        smiles = input(f"What is the smiles code of {mat._name}? "),
        name = mat._name,
        molar_mass = mat.molar_mass,
        phase=mat.phase,
        dissolves_in = set(a for a in input(f"What does {mat._name} dissolve in? ").split()),
        polarity = mat.polarity,
        dissolved_components = {input(f"{k} smiles? "):v for k,v in mat.dissolve().items()},
        boiling_point_K = mat._boiling_point,
        melting_point_K = mat._melting_point,
        specific_heat_J_gK = mat._specific_heat,
        enthalpy_fusion_J_mol = mat._enthalpy_fusion,
        enthalpy_vapor_J_mol = mat._enthalpy_vapor,
        density_g_ml = mat._density,
        spectra_overlap = mat.get_spectra_overlap(),
        spectra_no_overlap = mat.get_spectra_no_overlap(),
        color = mat._color,
    )

    return MaterialConstructor(kwargs)

MATERIALS = pd.read_pickle(os.path.dirname(__file__)+"/materials.pkl")

_matloc = MATERIALS.iloc
for i in range(MATERIALS.shape[0]):
    register({k:v for k,v in _matloc[i].items()})

if __name__ == "__main__":
    from chemistrylab.material0 import REGISTRY as oldreg
    
    print(MATERIALS)
    materials_list = []
    for key,matclass in oldreg.items():
        mat = matclass()
        args = from_old_mat(mat).get_args()
        print(args)
        materials_list.append(args)

    materials_df = pd.DataFrame(data=materials_list, columns=construct_signature)
    materials_df.to_pickle("materials.pkl")

