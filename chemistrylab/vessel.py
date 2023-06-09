from typing import NamedTuple, Tuple, Callable, Optional, List
import numpy as np
import numba
import pandas as pd
from chemistrylab.extract_algorithms import separate#separate_cc as separate

class Event(NamedTuple):
    name: str
    parameter: tuple
    other_vessel: Optional[object]

#Apparently documenting this isn't trivial :(
Event.name.__doc__ = "The registered name of the event function."
Event.parameter.__doc__ = "The parameters of the registered event function"
Event.other_vessel.__doc__ = "The other vessel needed for this event if requred (ex the target vessel when pouring)."


def _rebuild_solute_dict(solvent_dict, solute_dict, solvents):
    """
    Recreates the solute and solvent dict in the case where the solvents have changed.
    Args:
        solvent_dict (dict): The old solvent dict of (key,index) pairs
        solute_dict (dict): The old solute dict of (key,arr) pairs
        solvents (dict): The new list of solvents
    Returns:
        new_solvent_dict (dict): New set of (key,index) pairs
        new_solute_dict (dict): New set of (key,arr) pairs
    
    Note: This can be called less frequently if you allow solvents with zero amount to persist
    in the materials dict.
    """
    # needs to be called when adding or removing solvents (after you set the
    # mols dissolved in any removed solvents to 0)
    new_solvent_dict = {mat:i for i,mat in enumerate(solvents)}
    new_solute_dict = \
    {mat: np.array([solute_dict[mat][solvent_dict[sol]] if sol in solvent_dict else 0 
        for sol in solvents], dtype=np.float32) 
            for mat in solute_dict}
    return new_solvent_dict, new_solute_dict

@numba.jit(nopython=True)
def _validate_solute_amounts(mol_solute, mol_solvent, mol_dissolved):
    """
    Performs a series of consistency checks on the mol_dissolved array.

    Args:
        mol_solute (array): The total amount of each solute in the vessel (1D, size N)
        mol_solvent (array): The total amount of each solvent in the vessel (1D, size M)
        mol_dissolved (array) The amount of solute dissolved in each solvent (2D, shape [N,M])
    
    Checks:
    - Make sure the sum of each row in mol_dissolved is equal to the value in mol_solute
    - Strategies: Increase proportional to how much solvent there is
                      Decrease proportional to how much is already dissolved
    - Make sure each column of mol_dissolved if the corresponding value in mol_solvent is 0
    - Strategies: Set columns to 0 before doing the sum check if there is no solvent
    """
    # First make sure there are solvents
    tot_solvent=mol_solvent.sum()
    if tot_solvent<1e-12:
        mol_dissolved[:]=0
        return
    #get normalized solvent amounts
    norm_solvent=mol_solvent/tot_solvent
    #Make sure there are no solutes in empty solvents
    for j,v_mol in enumerate(mol_solvent):
        if v_mol<1e-12:
            mol_dissolved[:,j]=0
    #make sure the sums consistent
    for i,u_mol in enumerate(mol_solute):
        checksum=mol_dissolved[i].sum()
        #Set to zero if very close
        if u_mol<1e-16:
            mol_dissolved[i]=0
        #Decrease amounts proportional to what's in each solvent
        elif checksum>u_mol:
            mol_dissolved[i] *= (u_mol/checksum)
        #But increase amounts proportional to how much solvent there is
        elif checksum<u_mol:
            mol_dissolved[i] += (u_mol-checksum)*norm_solvent


layer_values=np.linspace(0, 1, 100, endpoint=True, dtype=np.float32)-1.9e-2

class Vessel:
    """
    The Vessel class serves as any container you might find in a lab, a beaker, a dripper, etc. 
    It simulates actions performed within a lab, such as draining contents, performing reactions, 
    mixing, pouring, etc. 

    The most important method is :meth:`~Vessel.push_event_to_queue` . The rest of the functions are
    either handeled in the backend or getter methods.

    These are the default :class:`Event` functions:

    +------------------+-----------------------------------------------------+-----------------------------------------+
    | Event Name       | Event Description                                   | Event Parameters                        |
    +==================+=====================================================+=========================================+
    |  pour by volume  | Pour from self vessel to target vessel by certain   |   volume (:class:`python:float`)        |
    |                  | volume                                              |                                         |
    +------------------+-----------------------------------------------------+-----------------------------------------+
    |  pour by percent | Pour a fraction of all contents in one vessel into  |   fraction (:class:`python:float`)      |
    |                  | another                                             |                                         |
    +------------------+-----------------------------------------------------+-----------------------------------------+
    |  drain by pixel  | Drain from self vessel to target vessel by certain  |   n_pixel (:class:`python:int`)         |
    |                  | pixel                                               |                                         |
    +------------------+-----------------------------------------------------+-----------------------------------------+
    |  mix             | Shake the vessel or let it settle                   |   t (:class:`python:float`)             |
    +------------------+-----------------------------------------------------+-----------------------------------------+
    |  update_layer    | Update self vessel's layer representation           |                                         |
    +------------------+-----------------------------------------------------+-----------------------------------------+
    |  change heat     | Add or remove heat from the vessel                  |   dQ (:class:`python:float`)            |
    +------------------+-----------------------------------------------------+-----------------------------------------+
    |  heat contact    | Connect the vessel to a reservoir for heat transfer | Tf (float), ht (float)                  |
    +------------------+-----------------------------------------------------+-----------------------------------------+

    """

    def __init__(
            self, 
            label: str, 
            temperature: float = 297.0, 
            volume: float = 1.0, 
            ignore_layout: bool = False
        ):
        """
        Args:
            label (str): Name for the vessel
            temperature (float): Temperature of the vessel in Kelvin
            volume (float): Volume of the vessel in Litres
        """
        #Functions to implement
        self.label=label
        self.default_dt=0.01
        self.temperature=temperature
        self.volume=volume
        self.material_dict=dict() # String keys, Material values
        self.solute_dict=dict() # String Keys, float array values
        self.solvent_dict=dict() #String Keys, index values
        self.solvents=[]
        self._layers_position = np.zeros(1, dtype=np.float32)
        self._layers_variance = np.array([self.volume/3.46], dtype=np.float32)
        self._layer_volumes = np.array([self.volume], dtype=np.float32)
        self._variance = 1e-5
        self._layers = None
        self.ignore_layout=ignore_layout
        self._layer_mats=[]

    def __repr__(self):
        return self.label

    def validate_solutes(self, checksum: bool = True):
        """
        Turns the solute dict into a 2D array, gets a 1D array
        of solute amounts, as well as a 1D array of solvent amounts, then
        performs consistency checks with _validate_solute_amounts.
        """
        if self.ignore_layout:return
        n_solvents=len(self.solvents)
        #gather a list of solutes
        solutes = tuple(a for a in self.material_dict if self.material_dict[a].is_solute())
        if n_solvents==0 or len(solutes)==0:return
        #get mol information for solutes and solvents
        solute_mols = np.array([self.material_dict[key].mol for key in solutes])
        solvent_mols = np.array([self.material_dict[key].mol for key in self.solvents])

        #get all of the dissolved amounts
        null = np.zeros(n_solvents)
        old_dict=self.solute_dict
        mol_dissolved=np.stack([old_dict[key] if key in old_dict else null for key in solutes])
        #run the validation
        _validate_solute_amounts(solute_mols, solvent_mols, mol_dissolved)
        #set the new solute dict
        self.solute_dict={key:mol_dissolved[i] for i,key in enumerate(solutes)}

    def validate_solvents(self):
        """
        Updates the solute_dict and solvent_dict / solvent array when the solvents have changed
        
        """
        if self.ignore_layout:return
        new_solvents = tuple(a for a in self.material_dict if self.material_dict[a].is_solvent())
        #union should be the same length as both new_solvents and solvents
        union = tuple(a for a in new_solvents if a in self.solvent_dict)
        if len(new_solvents)!=len(self.solvents) or len(new_solvents)!= len(union) :

            #copy over variances
            self._layers_variance = np.array([self._layers_variance[self.solvent_dict[sol]]
            if sol in self.solvent_dict else 0 for sol in new_solvents]+[self._layers_variance[-1]],
            dtype = np.float32)
            #copy over amounts
            self._layer_volumes = np.array([self._layer_volumes[self.solvent_dict[sol]]
            if sol in self.solvent_dict else 0 for sol in new_solvents]+[self._layer_volumes[-1]],
            dtype = np.float32)
            #make a new solvent dict
            self.solvent_dict,self.solute_dict = _rebuild_solute_dict(
                self.solvent_dict, self.solute_dict, new_solvents
            )
            
            n_solvents=len(new_solvents)
            # If we went from 0 to 1+ solvents we need to dissolve the solutes
            if len(self.solvents)==0 and n_solvents:
                for key,arr in self.solute_dict.items():
                    arr+=self.material_dict[key].mol/n_solvents
                
            self.solvents=new_solvents
            #last entry is for air
            self._layers_position = np.zeros(n_solvents+1, dtype=np.float32)

    def _handle_overflow(self):
        """
        Uses flled_volume() to determine if there is an overflow, if yes, it dumps a proportion
        of the vessel contents in order to have filled_volume <= volume.
        """
        filled = self.filled_volume()
        # decrease everything proportionally  and return -1 if there is an overflow
        if filled>self.volume:
            ratio = self.volume/filled
            for key,mat in self.material_dict.items():
                mat.mol*=ratio
            for key,arr in self.solute_dict.items():
                arr*=ratio
            return -1
        return 0

    def heat_capacity(self):
        """
        Returns:
            float: The sum of the heat capacities of all materials in the vessel (in J/K)
        """
        C_air = 1.2292875 #Heat capacity of air in J/L*K (near STP)
        #Adding in an approximate heat capacity for the air
        return self.volume*C_air+sum(mat.heat_capacity for a,mat in self.material_dict.items())

    def filled_volume(self):
        """
        Returns:
            float: The volume of all non-gas phase materials in the vessel (in Litres).
        """
        return sum(mat.litres for a,mat in self.material_dict.items())

    def get_material_dataframe(self):
        """
        Returns:
            :class:`~pandas.DataFrame`: A DataFrame detailing all materials present in the Vessel.  
        """
        info_dict = {key:(mat.mol,mat.phase,mat.is_solute(),mat.is_solvent()) for key,mat in self.material_dict.items()}
    
        return pd.DataFrame.from_dict(info_dict, orient="index",columns = ["Amount","Phase","Solute","Solvent"])

    def get_solute_dataframe(self):
        """
        Returns:
            :class:`~pandas.DataFrame`: A [solutes, solvents] DataFrame detailing how much solute is dissolved in each solvent.  
        """
        return pd.DataFrame.from_dict(self.solute_dict, orient="index",columns = self.solvents)

    def get_layer_dataframe(self):
        """
        Returns:
            :class:`~pandas.DataFrame`: A DataFrame containing the layer information of the vessel.  
        """                    
        info_dict = {mat._name:(
            self._layers_volume[i],
            self._layers_position[i],
            self._lvar[i],
            self._layer_colors[i],) for i,mat in enumerate(self._layer_mats)}

        info_dict["air"] = (
            self._layers_volume[-1],
            self._layers_position[-1],
            self._lvar[-1],
            self._layer_colors[-1],)
        return pd.DataFrame.from_dict(info_dict, orient="index",columns = ["volume","position","variance","color"])

    def push_event_to_queue(
            self,
            events: Tuple[Event] = tuple(), 
            dt: float= 0,
            update_layers: bool = True,
        ) -> Tuple[int]:
        """
        This function calls a set of event functions in sequence specified by `events`, then returns
        a tuple of status codes (one for each event).

        Args:
            events (Tuple[Event]): The sequence of events to be executed.
            dt (float): The amount of time elapsed (defaults to 0).
            update_layers (bool): Whether or not to update layer information at the end of the queue.

        Returns:
            Tuple[int]: A sequence of status codes for each event. At the moment, 0 represents normal execution,
            and -1 represents an illegal state reached (like a vessel overflow or boiling an empty vessel).
        """
        event_dict = type(self)._event_dict
        status=[]
        for event in events:
            status.append(event_dict[event.name](self, dt, event.other_vessel, *event.parameter))
        if (not self.ignore_layout) and update_layers:
            self._mix(0,None,dt)
            self._update_layers(0,None)
        return status

    def _heat_contact(self, dt, other_vessel, Tf, ht) -> int:
        """
        Rough Estimate of heat transfer so we can simulate putting something on a bunson burner
        or in a water bath.
        Param:
        - Tf (Optional[float]): The temperature of the heat source
        - ht (float): heat transfer coeff multiplied by how long you have
        
        Note: When T is None, ht will be used as heat Q
        Steps:
            1. Calculate total heat capacity of the vessel
            2. Get T estimate from heat capacity equation & Newton's law of heat transfer
            3. While T estimate is above the lowest boiling point of your materials
                i. Subtract d_ht required to get to boiling point from ht
                ii. Get boiling enthalpy of your material and calculate how much will boil
                iii. Boil off material and subtract the necessary d_ht used from ht
                iv. Calculate new T estimate based off of your new ht value
            4. set vessel temperature to T

        """
        use_dQ = (Tf is None)
        if use_dQ:
            # Adding heat (dQ) is the same as linearly approximating the exponential and log functions and setting
            # The reservoir temperature to one unit above the current temperature (trust me)
            exp= lambda x:1+x
            ln = lambda x:x-1
            Tf = self.temperature+1
        else:
            ln=np.log
            exp=np.exp


        other_mats=other_vessel.material_dict
        # Estimated final temperature before taking boiling into account
        T = Tf+(self.temperature-Tf)*exp(-ht/self.heat_capacity())

        mdict=self.material_dict
        #case for changing the heat of an empty vessel
        total_mats = sum(mat.mol for key,mat in mdict.items())
        if total_mats<1e-12:
            self.temperature = T
            # -1 if placing an empty beaker on something hot
            return 0 if Tf<373 else -1
        #get sorted boiling points
        bp = sorted([(key,mdict[key]._boiling_point) for key in mdict],key = lambda x:x[1])

        key,boil=bp[0]
        i=0
        while T>boil and T>self.temperature:
            material = mdict[key]
            #amount of transfer-time required to reach boiling temperature
            ht += ln((Tf-boil)/(Tf-self.temperature))*self.heat_capacity() #i
            self.temperature = boil #i
            if use_dQ: Tf = self.temperature+1
            boil_enthalpy=material.vapour_enthalpy #ii
            # ht = dQ/(T_f-T_boil)
            ht_used=min(ht,boil_enthalpy/(Tf-boil)) #ii
            ht-=ht_used #iii
            # Move the boiled material (iii)
            fraction = (ht_used*(Tf-boil)/boil_enthalpy) if ht_used > 0 else 0
            if key in other_mats:
                d_mol = material.mol*fraction
                other_mats[key].mol += d_mol
                material.mol -= d_mol
            else:
                other_mats[key] = material.ration(fraction)

            #Heat capacity is called again since it should be lower now
            T = Tf+(self.temperature-Tf)*exp(-ht/self.heat_capacity())
            i+=1
            if i>=len(bp):break
            key,boil=bp[i]
        self.temperature=T
        
        self.validate_solutes()
        other_vessel.validate_solvents()
        other_vessel.validate_solutes()

        return other_vessel._handle_overflow()
        
    def _change_heat(self, dt, other_vessel, dQ) -> int:
        """
        Depricated function, consider using heat contact instead.
        """
        return self._heat_contact(dt, other_vessel, None, dQ)
    
    def _pour_by_percent(self, dt, other_vessel, fraction) -> int:
        """
        - Take each material in this vessel and transfer it's amount * fraction
            to other_vessel
        - Check for overflow at the end
        - Dumping in contents should mix other_vessel

        TODO: Consider making this call a theoretical _add_materials function
        """
        if fraction<1e-16:return 0
        fraction = np.clip(fraction,0,1)
        other_mats=other_vessel.material_dict
        #Iterate through each material in the vessel
        for key,mat in self.material_dict.items():
            if mat.is_solute() and len(self.solvents)>0:
                self.solute_dict[key] *= (1-fraction)
            # Update other vessel's material dict
            if key in other_mats:
                other_mats[key].mol+=mat.mol*fraction
                mat.mol *= (1-fraction)
            else:
                # Get new material with the same class as mat using ration
                other_mats[key] = mat.ration(fraction)

        # Rebuild the other vessel's solute dict if new solvents have been added
        other_vessel.validate_solvents()
        other_vessel.validate_solutes()
        return other_vessel._handle_overflow()

    def _pour_by_volume(self, dt, other_vessel, volume) -> int:
        """
        Pour the same as _pour_by_percent but the fraction is determined by how much volume you want
        to pour vs how much of the vessel is filled.
        """
        filled_volume=self.filled_volume()
        if filled_volume<1e-12: return 0
        fraction = np.clip(volume/filled_volume,0,1)
        return self._pour_by_percent(dt, other_vessel, fraction)
    
    def _drain_by_pixel(self, dt, other_vessel, n_pixel) -> int:
        """
        This uses layer information to drain out the bottom N layers
        For each solvent:
            i. Figure out how much solvent is in the bottom N pixels and drain that
            ii. Figure out how much of EACH solute is dissolved in that bit of solvent and drain those
        
        """
        if self.ignore_layout:return -2

        other_mats=other_vessel.material_dict

        drained_layers = self._hashed_layers[:n_pixel]
        tot_pixels=len(self._hashed_layers)
        for i,key in enumerate(self.solvents):
            #NOTE mat is the solvent material
            mat = self.material_dict[key]
            drained_volume = (drained_layers==i).sum()/tot_pixels
            if drained_volume<=1e-12:continue
            #how much solvent is drained (percent wise)
            fraction=np.clip(drained_volume/self._layers_volume[i],0,1)
            if mat.litres<1e-12:
                self.DEBUG=self._hashed_layers*1,self.solvents
            #drain out the solvent
            if key in other_mats:
                d_mol=mat.mol*fraction
                other_mats[key].mol+=d_mol
                mat.mol -= d_mol
            else:
                other_mats[key] = mat.ration(fraction)
            #drain the solutes
            for u_key in self.solute_dict:
                #NOTE u_mat is the solute material
                u_mat = self.material_dict[u_key]
                removed_amount = fraction*self.solute_dict[u_key][i]
                #update the solute dict properly
                self.solute_dict[u_key][i] -= removed_amount
                if removed_amount<=1e-12:continue
                #drain out the solute
                if u_key in other_mats:
                    other_mats[u_key].mol+=removed_amount
                    u_mat.mol -= removed_amount
                else:
                    other_mats[u_key] = u_mat.ration(removed_amount/u_mat.mol)
        other_vessel.validate_solvents()
        other_vessel.validate_solutes()
        return other_vessel._handle_overflow()

    def _mix(self, dt, other_vessel, t) -> int:
        """
        Realistically just a wrapper for separate.mix

        This updates the amounts dissolved, layer positions and layer variances  
        """
        if self.ignore_layout:return -2
        t=np.float32(t) #or replace dt
        # Make air layer properties
        d_air = 1.225 #in g/L
        c_air = 0.65 #chosen color of air

        #This is just to ensure the order of solutes does not change
        s_names = tuple(s for s in self.solute_dict)
        #Grab solute and solvent objects
        solutes = tuple(self.material_dict[s] for s in s_names)
        solvents = [self.material_dict[s] for s in self.solvents]
        #Get solvent volumes
        

        solute_flag = sum(mat.mol for mat in solvents)<=1e-12
        misc_mats = [mat for key,mat in self.material_dict.items() 
            if (not mat.is_solvent()) and (solute_flag or not mat.is_solute())]

        layer_mats=solvents+misc_mats

        layer_volume = [mat.litres for mat in layer_mats]

        self._layer_mats = layer_mats

        # Add air to the end ov the volume list s.t it fills the remainder of the vessel
        layer_volume = np.array(layer_volume+[self.volume - sum(layer_volume)], dtype=np.float32)
        layer_density = np.array([mat.get_density() for mat in layer_mats]+[d_air], dtype=np.float32)
        self._layer_colors = np.array([mat._color for mat in layer_mats]+[c_air], dtype=np.float32)
        #Exclude air since it's not a solvent?
        solvent_polarity = np.array([mat.polarity for mat in solvents], dtype=np.float32)

        #Get solute properties
        solute_polarity = np.array([mat.polarity for mat in solutes], dtype=np.float32)
        #Hopefully this is the correct 2D array shape
        if len(s_names)>0:
            solute_amount = np.stack([self.solute_dict[a] for a in s_names]).astype(np.float32)
        else:
            solute_amount=np.zeros([0,len(solvents)],dtype=np.float32)

        solute_svolume = np.array([mat.litres_per_mol for mat in solutes], dtype=np.float32)
        
        self._layers_position, self._layers_volume, self._layers_variance, self._variance, new_solute_amount, self._lvar = separate.mix(
            layer_volume,
            self._layer_volumes.astype(np.float32),
            solute_svolume,
            self._layers_position.astype(np.float32),
            self._layers_variance.astype(np.float32),
            np.float32(self._variance),
            layer_density,
            solute_polarity,
            solvent_polarity,
            solute_amount,
            t
        )

        self._layer_volumes = layer_volume

        for i,s in enumerate(s_names):
            self.solute_dict[s] = new_solute_amount[i]

        return 0
   
    def _update_layers(self, dt, other_vessel) -> int:

        """
        This wraps separate.map_to_state
        It's used to get a layer image as well as layer information
        TODO: Handle solutes having a volume
        """

        self._layers,self._hashed_layers = separate.map_to_state(
            self._layers_volume.astype(np.float32),
            self._layers_position.astype(np.float32),
            self._lvar.astype(np.float32),
            self._layer_colors,
            layer_values
        )

    def get_layers(self):
        """
        Returns:
            List[float]: The color of each vessel layer.
        """
        if self._layers is None:
            self._mix(0,None,0)
            self._update_layers(0,None)
        return self._layers

    @classmethod
    def register(self, func: Callable, name: str):
        """
        The method to register an event function which updates a vessel instance.

        Args:
            func (Callable[[Vessel, Tuple, Optional[Vessel]], int]): An event function which acts on one or two vessels.
            name (str): The name of the event function for registration.
        """
        if name in self._event_dict:
            raise Exception(f"Cannot register the same Event ({name}) Twice!")
        self._event_dict[name]=func
    
    #ANY SUBCLASSES SHOULD DEFINE THIS EXPLICITLY!!!
    _event_dict = {
            'pour by volume': _pour_by_volume,
            'pour by percent':_pour_by_percent,
            'drain by pixel': _drain_by_pixel,
            'mix': _mix,
            'update layer': _update_layers,
            'change heat': _change_heat,
            'heat contact': _heat_contact,
        }