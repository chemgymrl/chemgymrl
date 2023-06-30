import numpy as np
import numba
from scipy.integrate import solve_ivp
from chemistrylab import material,vessel
from typing import NamedTuple, Tuple, Callable, Optional, List

from chemistrylab.reactions.reaction_info import ReactInfo


def _get_amounts(materials: Tuple[str], vessel: vessel.Vessel):
    n=np.zeros(len(materials))
    for i,key in enumerate(materials):
        if key in vessel.material_dict:
            n[i] = vessel.material_dict[key].mol
        else:
            n[i] = 0
    return n

def _set_amounts(materials, solvents, material_classes, n, vessel):
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




@numba.jit(nopython=True)
def get_rates(stoich_coeff_arr, pre_exp_arr, activ_energy_arr, conc_coeff_arr, num_reagents, temp, conc):
    """
    Finds the rate of reaction :math:`\\frac{dy}{dt}`

    Args:
        num_reagents (int): The number of reactants involved in the reaction
        temp (float): The temperature of the reactions
        conc (float): The initial concentrations of the materials
        *_arr (np.array): See :class:`~chemistrylab.reactions.reaction_info.ReactInfo`

    Returns:
        np.array: Rates of change in concentration :math:`\\frac{dy}{dt}`.
    """

    
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

@numba.njit
def newton_solve(stoich_coeff_arr, pre_exp_arr, activ_energy_arr, conc_coeff_arr, num_reagents, temp, conc, dt, N):
    """

    Args:
        num_reagents (int): The number of reactants involved in the reaction
        temp (float): The temperature of the reactions
        conc (float): The initial concentrations of the materials
        dt (float): The amount of time to pass
        N (int): The minimum number of time-steps to break dt into
        *_arr (np.array): See :class:`~chemistrylab.reactions.reaction_info.ReactInfo`

    Returns:
        np.array: The final concentrations y(dt)

    Solves the initial value problem :math:`\\frac{dy}{dt} = f(y)` specifically in the case where f(y) is chemical a rate calculation
    
    (The problem is that we have :math:`y(t_0)` and need :math:`y(t_0+dt))`
    
    This solver uses newton's method:
        set :math:`T = \\frac{dt}{N}`

        set :math:`y_0 = y(t_0)`

        for n = 1,2,3,...N:
            :math:`y_n = y_{n-1} + f(y_{n-1})*T`
        
        :math:`y(dt) \\leftarrow y_N`
   
    Intuitively, it is like taking a Riemann sum of dy/dt (but you get dy/dt by bootstrapping your current sum for y(t))
    This implementation uses a variable step size in order to account for super fast-changing concentrations (wurtz distill)
    """
    R = 8.314462619
    #if your updates are below 5e-4 you can increase factor (I decided this is a good number)
    targ = 5e-4
    
    ddt=dt/N
    
    k = (ddt*pre_exp_arr) * np.exp((-1.0 * activ_energy_arr) / (R * temp))
    
    factor=1
    count=0
    d_conc=conc*0
    while dt>0:
        conc = np.clip(conc, 0, None)
        #k are the reaction constants
        
        rates = k*1
        for i in range(len(rates)):
            for j in range(num_reagents):
                rates[i] *= conc[j] ** stoich_coeff_arr[i][j]
        
        #calculate concentration changes
        for i in range(conc.shape[0]):
            d_conc[i]=0
            for j in range(rates.shape[0]):
                d_conc[i] += conc_coeff_arr[i][j]*rates[j]
        
        #maximum proportion of material reduction        
        ratio=np.max(-d_conc/(conc+1e-6))
        
        #mess with the step size to make sure you don't get any super huge concentration decreases
        while ratio*factor<targ and factor<10:
            factor*=2
        while ratio*factor>0.1:
            factor*=0.5
        if factor*ddt>=dt:
            factor = dt/ddt
            dt=0
        
        dt-=factor*ddt            
        count+=1
        # Add concentration changes
        conc+=d_conc*factor

    return conc
   

class Reaction():
    def __init__(self,react_info: ReactInfo, solver: str = 'RK45', newton_steps: int = 100):
        """

        A class to update concentrations of the materials in a vessel according to a reaction.

        Args:
            react_info (ReactInfo): Named Tuple containing all necessary reaction information
            solver (str): Which solver to use
            newton_steps (int): How many steps to use when the solver is 'newton'
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
        self.materials=react_info.MATERIALS
        self.material_classes = tuple(material.REGISTRY[key] for key in self.materials)
        
        #Necessary for calculating rates
        self.stoich_coeff_arr = react_info.stoich_coeff_arr
        self.pre_exp_arr = react_info.pre_exp_arr
        self.activ_energy_arr = react_info.activ_energy_arr
        self.conc_coeff_arr = react_info.conc_coeff_arr
        self.num_reagents = len(self.reactants)

    def update_concentrations(self,vessel: vessel.Vessel, dt: float = 0):
        """
        Takes in a vessel and applies the reaction to it, updating the material and solvent dicts in the process

        Args:
            vessel (Vessel): The vessel to perform the reaction on
            dt (float): The amount of time passed during the reaction

        """
        
        n = _get_amounts(self.materials, vessel)
        if n.sum() < 1e-12:return
        temperature = vessel.temperature
        current_volume = vessel.filled_volume()
        
        if dt==0:
            dt = vessel.default_dt
        
        #update concentrations
        new_n = self.react(n, temperature, current_volume, dt)
        #set the updated concentrations
        _set_amounts(self.materials, self.solvents, self.material_classes, new_n, vessel)
        
    def react(self, n: np.array, temp: float, volume: float, dt: float):
        """
        Args:
            n (np.array): An array containing the amounts of each material in the vessel.
            temp (float): The temperature of the system in Kelvin.
            volume (float): The volume of the system in Litres.
            dt (float): The time-step demarcating separate steps in seconds.
        Returns:
            np.array: The new amounts of each material in the vessel
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
    
    
NoneType = type(None)
def react(vessel: vessel.Vessel, dt: float, other_vessel: NoneType, reaction: Reaction):
    """
    :class:`~chemistrylab.vessel.Event` function to perform a reaction
    
    """
    reaction.update_concentrations(vessel , dt)
    return 0

vessel.Vessel.register(func = react, name = 'react')