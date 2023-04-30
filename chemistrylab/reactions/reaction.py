
def get_amounts(materials, vessel):
    n=np.zeros(len(materials))
    for i,key in enumerate(materials):
        if key in vessel.material_dict:
            n[i] = vessel.material_dict[key].mol
        else:
            n[i] = 0
    return n

def set_amounts(materials,solvents, material_classes, n, vessel):
    for i,key in enumerate(materials):
        amount = n[i]
        if key in vessel.material_dict:
            vessel.material_dict[key].mol=amount
        elif n[i]>0:
            mat = material_classes[i]()
            mat.mol=amount
            vessel.material_dict[key] = mat
    vessel.validate_solvents()
    vessel.validate_solutes()
        
#####################################################################################################################################
import numpy as np
import numba
from scipy.integrate import solve_ivp
from chemistrylab.chem_algorithms import material



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


@numba.jit
def newton_solve(stoich_coeff_arr, pre_exp_arr, activ_energy_arr, conc_coeff_arr, num_reagents, temp, conc, dt, N):
    """
    Solves the initial value problem dy/dt = f(y) specifically in the case where f(y) is chemical a rate calculation
    
    (The problem is that we have y(t0) and need y(t0+dt))
    
    This solver uses newton's method:
        set T = dt/N
        set y_0 = y(t0)
        for n = 1,2,3,...N:
            y_n = y_(n-1) + f(y_(n-1))*T
        
        y(dt) <- y_N 
   
    Intuitively, it is like taking a Riemann sum of dy/dt (but you get dy/dt by bootstrapping your current sum for y(t))
    """
    R = 8.314462619
    
    ddt=dt/N
    k = (ddt*pre_exp_arr) * np.exp((-1.0 * activ_energy_arr) / (R * temp))
        
    for n in range(N):
        conc = np.clip(conc, 0, None)
        #k are the reaction constants
        
        rates = k*1
        for i in range(len(rates)):
            for j in range(num_reagents):
                rates[i] *= conc[j] ** stoich_coeff_arr[i][j]
        #calculate concentration changes and add them to the concentration
        for i in range(conc.shape[0]):
            for j in range(rates.shape[0]):
                conc[i] += conc_coeff_arr[i][j]*rates[j]
                
    return conc
   

class Reaction():
    def __init__(self,react_info,solver='RK45',newton_steps=100):
        """
        Args:
        - react_info (ReactInfo): Named Tuple containing all necessary reaction information
        - solver (str): Which solver to use
        - newton_steps (int): How many steps to use when the solver is 'newton'
        """
        
        if not solver in {'RK45', 'RK23', 'DOP853', 'DBF', 'LSODA','newton'}:
            solver='RK45'
        self.solver=solver
        self.newton_steps=newton_steps
        #has to be set somewhere
        self.threshold=1e-12
        
        #materials we need for the reaction
        self.reactants=react_info.REACTANTS
        self.products=react_info.PRODUCTS
        self.solvents=react_info.SOLVENTS
        #Concatenate all of the materials (this should realistically be done in the reaction file since it has a direct
        #Impact on the conc_coeff_arr
        self.materials=[]
        for mat in self.reactants + self.products + self.solvents:
            if mat not in self.materials:
                self.materials.append(mat)
        self.material_classes = tuple(material.REGISTRY[key] for key in self.materials)
        
        #Necessary for calculating rates
        self.stoich_coeff_arr = react_info.stoich_coeff_arr
        self.pre_exp_arr = react_info.pre_exp_arr
        self.activ_energy_arr = react_info.activ_energy_arr
        self.conc_coeff_arr = react_info.conc_coeff_arr
        self.num_reagents = len(self.reactants)

    def update_concentrations(self,vessel):
        """
        Takes in a vessel and applies the reaction to it, updating the material and solvent dicts in the process
        """
        
        
        n = get_amounts(self.materials, vessel)
        if n.sum() < 1e-12:return
        temperature = vessel.temperature
        current_volume = vessel.filled_volume()
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
        Returns:
        - new_n (np.array): The new amounts of each material in the vessel
        """
        # set the intended vessel temperature in the differential equation module
        self.temp = temp
        conc=n/volume
        
        
        if self.solver=='newton':
            #newton solver should be faster but less accurate
            new_conc = newton_solve(self.stoich_coeff_arr, self.pre_exp_arr,
                         self.activ_energy_arr, self.conc_coeff_arr,
                         self.num_reagents, self.temp, conc, dt, self.newton_steps)
        else:
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
    
    
  