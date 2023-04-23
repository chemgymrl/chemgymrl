####################################Vessel-Reaction Translation functions (Will be redone)#########################################

def get_amounts(materials, material_dict):
    n=np.zeros(len(materials))
    null=(0,0)
    for i,mat in enumerate(materials):
        n[i]=material_dict.get(mat,null)[1]
    return n

#trash function which just gives a temporary go-between from the new reactions to the old vessels (will be rewritten or discarded)
def set_amounts(materials,solvents, material_classes, n, vessel):
     # tabulate all the materials and solutes used and their new values
    new_material_dict = vessel.get_material_dict()
    new_solute_dict = vessel.get_solute_dict()
    for i,material_name in enumerate(materials):
        
        amount = n[i]
        try:
            new_material_dict[material_name] = [new_material_dict[material_name][0], amount, 'mol']
        except KeyError:
            new_material_dict[material_name] = [material_classes[i](), amount, 'mol']
        if new_material_dict[material_name][0].is_solute():
            # create the new solute dictionary to be appended to a new vessel object
            solute_change_amount = new_material_dict[material_name][1] - vessel.get_material_amount(material_name)
            try:
                for solvent in new_solute_dict[material_name]:
                    if vessel._material_dict[material_name][1] > 1e-12:
                        new_solute_dict[material_name][solvent][1] += solute_change_amount * vessel._solute_dict[material_name][solvent][1] / vessel._material_dict[material_name][1]
            except KeyError:
                present_solvents = []
                for solvent in solvents:
                    if vessel.get_material_amount(solvent) > 1e-12:
                        present_solvents.append(solvent)

                new_solute_dict[material_name] = {solvent: [convert_to_class(materials=[solvent])[0](), solute_change_amount/len(solvents), 'mol'] for solvent in present_solvents}

    # update target vessel's material amount
    vessel._material_dict = new_material_dict
    # update target vessel's solute dict
    vessel._solute_dict = new_solute_dict
    # empty call to mix to rebuild layers?
    __ = vessel.push_event_to_queue(events=None,dt=0)

    
#####################################################################################################################################
import numpy as np
import numba
from scipy.integrate import solve_ivp
from chemistrylab.reactions.get_reactions import convert_to_class



@numba.jit
def get_rates(stoich_coeff_arr, pre_exp_arr, activ_energy_arr, conc_coeff_arr, num_reagents, temp, conc):
    R = 8.314462619
    
    conc = np.clip(conc, 0, None)
    #k are the reaction constants
    k = pre_exp_arr * np.exp((-1.0 * activ_energy_arr) / (R * temp))
        
    rates = k*1
    for i in range(len(rates)):
        for j in range(num_reagents):
            rates[i] *= conc[j] ** stoich_coeff_arr[i][j]
    
    conc_change = np.zeros(conc_coeff_arr.shape[0])
    for i in range(conc_change.shape[0]):
        for j in range(rates.shape[0]):
            conc_change[i] += conc_coeff_arr[i][j]*rates[j]
    
    return conc_change
   

class Reaction():
    def __init__(self,react_module,solver='RK45'):
        
        if not solver in {'RK45', 'RK23', 'DOP853', 'DBF', 'LSODA'}:
            solver='RK45'
        self.solver=solver
        
        #has to be set somewhere
        self.threshold=1e-12
        
        #materials we need for the reaction
        self.reactants=react_module.REACTANTS
        self.products=react_module.PRODUCTS
        self.solvents=react_module.SOLVENTS
        #Concatenate all of the materials (this should realistically be done in the reaction file since it has a direct
        #Impact on the conc_coeff_arr
        self.materials=[]
        for mat in self.reactants + self.products + self.solvents:
            if mat not in self.materials:
                self.materials.append(mat)
        self.material_classes = convert_to_class(materials=self.materials)
        
        #Necessary for calculating rates
        self.stoich_coeff_arr = react_module.stoich_coeff_arr
        self.pre_exp_arr = react_module.pre_exp_arr
        self.activ_energy_arr = react_module.activ_energy_arr
        self.conc_coeff_arr = react_module.conc_coeff_arr
        self.num_reagents = len(self.reactants)

    def update_concentrations(self,vessel):
        """
        Takes in a vessel and applies the reaction to it, updating the material and solvent dicts in the process
        """
        n = get_amounts(self.materials, vessel._material_dict)
        temperature = vessel.get_temperature()
        current_volume = vessel.get_current_volume()[-1]
        dt = vessel.default_dt
        
        #update concentrations
        new_n = self.react(n, temperature, current_volume, dt)
        #set the updated concentrations
        set_amounts(self.materials, self.solvents, self.material_classes, new_n, vessel)
        
    def react(self, n, temp, volume, dt):
        """
        Args:
        - n (np.array): An array containing the amounts of each material in the vessel.
        - temp (float): The temperature of the system in Kelvin.
        - volume (float): The volume of the system in Litres.
        - dt (float): The time-step demarcating separate steps in seconds.
        Returns
        - new_n (np.array) The new amounts of each material in the vessel
        """
        # set the intended vessel temperature in the differential equation module
        self.temp = temp
        conc=n/volume
        new_conc = solve_ivp(self, (0, dt), conc, method=self.solver).y[:, -1]
        new_n = new_conc * volume
        #set negligible amounts to 0
        new_n *= (new_n > self.threshold)
        
        return new_n
    
    def __call__(self, t, conc):
        """
        a function that calculates the change in concentration given a current concentration
        remember to set the temperature before you call this function
        This function is mainly used with the scipy ODE solvers
        """
        return get_rates(self.stoich_coeff_arr, self.pre_exp_arr,
                         self.activ_energy_arr, self.conc_coeff_arr,
                         self.num_reagents, self.temp, conc)
    
    
  