from typing import NamedTuple, Tuple, Callable, Optional
import numpy as np
import numba
from chemistrylab.extract_algorithms import separate

class Event(NamedTuple):
    name: str
    parameter: tuple
    other_vessel: Optional[object]

def rebuild_solute_dict(solvent_dict, solute_dict, solvents):
    """
    Recreates the solute and solvent dict in the case where the solvents have changed.
    Args:
    - solvent_dict (dict): The old solvent dict of (key,index) pairs
    - solute_dict (dict): The old solute dict of (key,arr) pairs
    - solvents (dict): The new list of solvents
    Returns:
    - new_solvent_dict (dict): New set of (key,index) pairs
    - new_solute_dict (dict): New set of (key,arr) pairs
    
    Note: This can be called less frequently if you allow solvents with zero amount to persist
    in the materials dict.
    """
    # needs to be called when adding or removing solvents (after you set the
    # mols dissolved in any removed solvents to 0)
    new_solvent_dict = {mat:i for i,mat in enumerate(solvents)}
    new_solute_dict = \
    {mat: np.array([solute_dict[mat][solvent_dict[sol]] if sol in solvent_dict else 0 
        for sol in solvents]) 
            for mat in solute_dict}
    return new_solvent_dict, new_solute_dict

@numba.jit
def validate_solute_amounts(mol_solute, mol_solvent, mol_dissolved):
    """
    Performs a series of consistency checks on the mol_dissolved array.

    Args:
    - mol_solute (array): The total amount of each solute in the vessel (1D, size N)
    - mol_solvent (array): The total amount of each solvent in the vessel (1D, size M)
    - mol_dissolved (array) The amount of solute dissolved in each solvent (2D, shape [N,M])
    
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


class Vessel:
    """
    Class defining the Vessel object.
    """

    def __init__(
            self,
            label,
            temperature=297.0,
            volume=1.0,
            ignore_layout=False
        ):
        """
        Args:
        - label (str): Name for the vessel
        - temperature (float): Temperature of the vessel in Kelvin
        - volume (float): Volume of the vessel in Litres
        
        """
        #Functions to implement
        self._event_dict = {
            'pour by volume': self._pour_by_volume,
            'dump fraction':self._dump_fraction,
            'drain by pixel': self._drain_by_pixel,
            'mix': self._mix,
            'update layer': self._update_layers,
            'change heat': self._change_heat,
            'heat contact': self._heat_contact,
            #The ones below are in the original vessel but I'm not convinced they should
            # exist/do the same thing
            #'update material dict': self._update_material_dict,
            #'update solute dict': self._update_solute_dict,
            #'update temperature': self._update_temperature,
            #'wait': self._wait
        }
        self.default_dt=0.01
        self.temperature=temperature
        self.volume=volume
        self.material_dict=dict() # String keys, Material values
        self.solute_dict=dict() # String Keys, float array values
        self.solvent_dict=dict() #String Keys, index values
        self.solvents=[]
        self._layers_position = np.zeros(1)
        self._layers_variance = np.ones(1)*self.volume/3.46
        self._solvent_volumes = np.ones(1)*self.volume
        self._variance = 2
        self._layers = None
        self.ignore_layout=ignore_layout

    def validate_solutes(self,checksum=True):
        """
        Turns the solute dict into a 2D array, gets a 1D array
        of solute amounts, as well as a 1D array of solvent amounts, then
        performs consistency checks with validate_solute_amounts.
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
        validate_solute_amounts(solute_mols, solvent_mols, mol_dissolved)
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
            if sol in self.solvent_dict else 0 for sol in new_solvents]+[self._layers_variance[-1]])
            #copy over amounts
            self._solvent_volumes = np.array([self._solvent_volumes[self.solvent_dict[sol]]
            if sol in self.solvent_dict else 0 for sol in new_solvents]+[self._solvent_volumes[-1]])
            #make a new solvent dict
            self.solvent_dict,self.solute_dict = rebuild_solute_dict(
                self.solvent_dict, self.solute_dict, new_solvents
            )
            
            n_solvents=len(new_solvents)
            # If we went from 0 to 1+ solvents we need to dissolve the solutes
            if len(self.solvents)==0 and n_solvents:
                for key,arr in self.solute_dict.items():
                    arr+=self.material_dict[key].mol/n_solvents
                
            self.solvents=new_solvents
            #last entry is for air
            self._layers_position = np.zeros(n_solvents+1)

    def handle_overflow(self):
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
        C_air = 1.2292875 #Heat capacity of air in J/L*K (near STP)
        #Adding in an approximate heat capacity for the air
        return self.volume*C_air+sum(mat.heat_capacity for a,mat in self.material_dict.items())

    def filled_volume(self):
        return sum(mat.litres for a,mat in self.material_dict.items())

    def push_event_to_queue(
            self,
            events: Tuple[Event] = tuple(), 
            dt: Optional[float] = None,
        ) -> Tuple[int]:
        # I intend the int returned to be something of a status code
        # Ex: 0 could be regular operation, -1 is a spill, etc
        # This function is set to call a set of event functions in sequence specified by `events`
        # It would then return some status code corresponding to the completed events
        out=[]
        if dt is None:dt=0
        for event in events:
            if event.other_vessel!=None:
                out.append(self._event_dict[event.name](event.parameter,event.other_vessel,dt))
            else:
                out.append(self._event_dict[event.name](event.parameter,dt))

        if not self.ignore_layout:
            self._mix((0,),0)
            self._update_layers(0,0)
        return out

    def _heat_contact(self,param,other_vessel,dt):
        """
        Rough Estimate of heat transfer so we can simulate putting something on a bunson burner
        or in a water bath.
        Param:
        - T (float): The temperature of the heat source
        - ht (float): heat transfer coeff multiplied by how long you have
        """
        Tf,ht=param

        other_mats=other_vessel.material_dict
        # Estimated final temperature taking boiling into account
        T = Tf+(self.temperature-Tf)*np.exp(-ht/self.heat_capacity())

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
            ht -= np.log((Tf-self.temperature)/(Tf-boil))*self.heat_capacity() #i
            self.temperature = boil #i
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
            T = Tf+(self.temperature-Tf)*np.exp(-ht/self.heat_capacity())
            i+=1
            if i>=len(bp):break
            key,boil=bp[i]
        self.temperature=T
        
        self.validate_solutes()
        other_vessel.validate_solvents()
        other_vessel.validate_solutes()

        return other_vessel.handle_overflow()
        
    def _change_heat(self, param, other_vessel, dt) -> int:
        
        """
        Heat Capacity Equation:
        dQ = sum(Ci)dT
        Here Ci is the heat capacity of all of your materials

        Steps:
            1. Calculate total heat capacity of the vessel
            2. Get dT from heat capacity equation
            3. While T+dT is above the lowest boiling point of your materials
            i. Subtract Q required to get to boiling point from dQ
            ii. Get boiling enthalpy of your material and calculate how much will boil
            iii. Boil off material and subtract the necessary Q used from dQ
            iv. Calculate new dT based off of your new dQ value
            4. T+=dT
            return
        """
        dQ=param[0]
        other_mats=other_vessel.material_dict
        dT = dQ/self.heat_capacity()
        mdict=self.material_dict

        #case for changing the heat of an empty vessel
        total_mats = sum(mat.mol for key,mat in mdict.items())
        if total_mats<1e-12:
            self.temperature+=dT
            return 0 if dT<=0 else -1
        #get sorted boiling points
        bp = sorted([(key,mdict[key]._boiling_point) for key in mdict],key = lambda x:x[1])

        key,boil=bp[0]
        i=0
        while self.temperature+dT>boil:
            material = mdict[key]
            dQ -= (boil-self.temperature)*self.heat_capacity() #i
            self.temperature = boil #i
            boil_enthalpy=material.vapour_enthalpy #ii
            Q_used=min(dQ,boil_enthalpy) #ii
            dQ-=Q_used #iii
            # Move the boiled material (iii)
            fraction = (Q_used/boil_enthalpy) if Q_used > 0 else 0
            if key in other_mats:
                d_mol = material.mol*fraction
                other_mats[key].mol += d_mol
                material.mol -= d_mol
            else:
                other_mats[key] = material.ration(fraction)
            #Heat capacity is called again since it should be lower now
            dT = dQ/self.heat_capacity() #iv
            
            i+=1
            if i>=len(bp):break
            key,boil=bp[i]
        self.temperature+=dT
        
        self.validate_solutes()
        other_vessel.validate_solvents()
        other_vessel.validate_solutes()

        return other_vessel.handle_overflow()
    
    def _dump_fraction(self, param, other_vessel, dt ) -> int:
        """
        - Take each material in this vessel and transfer it's amount * fraction
            to other_vessel
        - Check for overflow at the end
        - Dumping in contents should mix other_vessel

        TODO: Consider making this call a theoretical _add_materials function
        """
        fraction = np.clip(param[0],0,1)
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
        return other_vessel.handle_overflow()

    def _pour_by_volume(self, param, other_vessel, dt) -> int:
        """
        Pour the same as _dump_fraction but the fraction is determined by how much volume you want
        to pour vs how much of the vessel is filled.
        """
        volume=param[0]
        filled_volume=self.filled_volume()
        if filled_volume<1e-12: return 0
        fraction = np.clip(volume/filled_volume,0,1)
        return self._dump_fraction([fraction], other_vessel, dt)
    
    def _drain_by_pixel(self, param, other_vessel, dt) -> int:
        """
        This uses layer information to drain out the bottom N layers
        For each solvent:
            i. Figure out how much solvent is in the bottom N pixels and drain that
            ii. Figure out how much of EACH solute is dissolved in that bit of solvent and drain those
        
        """
        if self.ignore_layout:return -2

        n_pixel = param[0]

        other_mats=other_vessel.material_dict

        drained_layers = self._hashed_layers[:n_pixel]
        tot_pixels=len(self._hashed_layers)
        for i,key in enumerate(self.solvents):
            #NOTE mat is the solvent material
            mat = self.material_dict[key]
            drained_volume = (drained_layers==i).sum()/tot_pixels
            if drained_volume<=1e-12:continue
            #how much solvent is drained (percent wise)
            fraction=np.clip(drained_volume/mat.litres,0,1)
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
        return other_vessel.handle_overflow()

    def _mix(self, param, dt) -> int:
        """
        Realistically just a wrapper for separate.mix

        This updates the amounts dissolved, layer positions and layer variances  
        """
        if self.ignore_layout:return -2
        t=param[0] #or replace dt
        # Make air layer properties
        d_air = 1.225 #in g/L

        #This is just to ensure the order of solutes does not change
        s_names = tuple(s for s in self.solute_dict)
        #Grab solute and solvent objects
        solutes = tuple(self.material_dict[s] for s in s_names)
        solvents = tuple(self.material_dict[s] for s in self.solvents)
        #Get solvent volumes
        solvent_volume = [mat.litres for mat in solvents]
        # Add air to the end ov the volume list s.t it fills the remainder of the vessel
        solvent_volume = np.array(solvent_volume+[self.volume - sum(solvent_volume)])
        solvent_density = np.array([mat.get_density() for mat in solvents]+[d_air])
        #Exclude air I guess?
        solvent_polarity = np.array([mat.polarity for mat in solvents])

        #Get solute properties
        solute_polarity = np.array([mat.polarity for mat in solutes])
        #Hopefully this is the correct 2D array shape
        if len(s_names)>0:
            solute_amount = np.stack([self.solute_dict[a] for a in s_names])
        else:
            solute_amount=np.zeros([0,len(solvents)])
        
        self._layers_position, self._layers_variance, self._variance, new_solute_amount, __ = separate.mix(
            v=solvent_volume,
            Vprev = self._solvent_volumes,
            B=self._layers_position,
            C=self._layers_variance,
            C0=self._variance,
            D=solvent_density,
            Spol=solute_polarity,
            Lpol=solvent_polarity,
            S=solute_amount,
            mixing=t
        )

        self._solvent_volumes = solvent_volume

        for i,s in enumerate(s_names):
            self.solute_dict[s] = new_solute_amount[i]
        
        return 0
   
    def _update_layers(self, param, dt) -> int:

        """
        This wraps separate.map_to_state
        It's used to get a layer image as well as layer information
        TODO: Handle solutes having a volume
        """
        _=param
        v_air = self.volume-self.filled_volume()
        c_air = 0.65 #chosen color of air
        solvents = tuple(self.material_dict[s] for s in self.solvents)
        solvent_volume = [mat.litres for mat in solvents]
        # Add air to the end ov the volume list s.t it fills the remainder of the vessel
        solvent_volume = np.array(solvent_volume+[self.volume - sum(solvent_volume)])

        solvent_colors = np.array([mat._color for mat in solvents]+[c_air])

        self._layers,self._hashed_layers = separate.map_to_state(
            A=solvent_volume,
            B=self._layers_position,
            C=self._layers_variance,
            colors=solvent_colors,
            x=separate.x
        )

    def get_layers(self):
        if self._layers is None:self._update_layers(0,0)
        return self._layers