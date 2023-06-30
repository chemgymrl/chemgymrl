import numpy as np
import gymnasium as gym
from copy import deepcopy

def get_reaction(env):
    try:
        return env.default_events[0].parameter[0]
    except:
        return None
    
    
def prep_no_overflow(env):
    """Increases the capacity of each vessel so overflow can't occur and deepcopies the initial set of vessels"""
    tot_vol = sum(v.filled_volume() for v in env.shelf)
    tot_vol = 2**np.ceil(np.log(tot_vol)/np.log(2))
    for v in env.shelf:
        v.volume=tot_vol
    #collect initial vessels  
    start_vessels = deepcopy([v for v in env.shelf])
    return start_vessels
    
def run_env_no_overflow(env_id,seed=1,acts = None, trace = False):
    """
    Runs a chemgymrl env while making sure no overflow happens in the vessels.
    
    """
    if acts is None:
        acts = []
    env = gym.make(env_id)
    _ = env.reset(seed=seed)
    
    #increase vessel sizes so no overflow happens
    tot_vol = sum(v.filled_volume() for v in env.shelf)
    tot_vol = 2**np.ceil(np.log(tot_vol)/np.log(2))
    for v in env.shelf:
        v.volume=tot_vol
    #collect initial vessels  
    start_vessels = deepcopy([v for v in env.shelf])    
    d = False
    i=0
    while not d:
        if len(acts)<=i:
            act = env.action_space.sample()
            acts+=[act]
        else:
            act=acts[i]
        o,r,d,*_ = env.step(act)
        i+=1
    if trace:
        return start_vessels,env.shelf.get_vessels(), get_reaction(env), env, acts
    return start_vessels,env.shelf.get_vessels(), get_reaction(env)



def chemgym_filter(x):
    y=[]
    for env in x:
        if ("React" in env or "Extract" in env or "Distill" in env) and not "Discrete" in env:
            if not "Demo" in env and not "v0" in env:
                y+=[env]
    return y


def check_conservation(start_vessels,end_vessels, TOL = 1e-8):
    """
    Intended use: 
    1. Disable overflow on all vessels
    2. Do a deepcopy of the vessels list
    3. Run your test code on the vessels
    4. Use this function to make sure conservation laws apply
    
    Args:
    - start_vessels (Tuple[Vessel]): A copy of the initial vessels list
    - end_vessels (Tuple[Vessel]): The final vessel list
    
    
    Performs a checksum on each  material in each set of vessels to ensure conservation of materials.
    """
    start_total = dict()
    end_total = dict()
    for vessel in start_vessels:
        # Make sure get_material_dataframe is always implemented no matter how data is stored internally
        mat_df = vessel.get_material_dataframe()
        #set the total amount of each material in starting vessels
        for name,data in mat_df.iterrows():
            start_total[name] = start_total.get(name,0)+data.Amount
    for vessel in end_vessels:
        mat_df = vessel.get_material_dataframe()
        #set the total amount of each material in ending vessels
        for name,data in mat_df.iterrows():
            end_total[name] = end_total.get(name,0)+data.Amount
    #Making sure all of the materials still exist at the end
    for mat in start_total:
        if not mat in end_total:
            return False
    #perform the checksum
    for mat,amount in end_total.items():
        if not mat in start_total:
            return False
        if abs(amount-start_total[mat])>TOL:
            print(amount,start_total[mat])
            return False
    return True

def check_non_negative(vessels):
    """
    Checks that both materials and dissolved amounts are non-negative
    """

    for vessel in vessels:
        mat_df = vessel.get_material_dataframe()
        #set the total amount of each material in ending vessels
        for name,data in mat_df.iterrows():
            if data.Amount<0:
                return False
        sol_df = vessel.get_solute_dataframe()
        for col,data in sol_df.iterrows():
            for dissolved in data:
                if dissolved<0:
                    return False
    return True



def get_diff(vectors,x):
    """
    Computes the difference between x and the projection of x into the space spanned by 'vectors'.
    """
    # Q contains an orthogonal set of unit vectors spanning your vectors
    # Note Q has extra vectors that makes it span Rn, but for the extra vectors, the row in r will be zero
    q,r = np.linalg.qr(vectors)
    orthog_set = [v for i,v in enumerate(q.T) if abs(r[i][i])>1e-8]
    projection = np.sum([v*np.dot(v,x) for v in orthog_set],axis=0)
    return x-projection


def check_conservation_react(start_vessels,end_vessels, reaction, TOL = 1e-8):
    """
    Intended use: 
    1. Disable overflow on all vessels
    2. Do a deepcopy of the vessels list
    3. Run your test code on the vessels
    4. Use this function to make sure conservation laws apply
    
    Args:
    - start_vessels (Tuple[Vessel]): A copy of the initial vessels list
    - end_vessels (Tuple[Vessel]): The final vessel list
    - react_info (ReactInfo): Reaction info necessary to account for materials transforming into other materials
    
    
    Performs a checksum on each  material in each set of vessels to ensure conservation of materials.
    This accounts for reactions by using QR decomposition to get an orthogonal set of vectors spanning the space of all 
    possible concentration changes, then making sure the change in concentrations from initial to final vessel sets
    is inside this space via projection.
    """
    
    reactants = {mat:i for i,mat in enumerate(reaction.materials)}
    
    start_react = np.zeros(len(reactants))
    
    end_react = np.zeros(len(reactants))
    
    start_total = dict()
    end_total = dict()
    for vessel in start_vessels:
        # Make sure get_material_dataframe is always implemented no matter how data is stored internally
        mat_df = vessel.get_material_dataframe()
        #set the total amount of each material in starting vessels
        for name,data in mat_df.iterrows():
            if name in reactants:
                start_react[reactants[name]]+=data.Amount
            else:
                start_total[name] = start_total.get(name,0)+data.Amount
    for vessel in end_vessels:
        mat_df = vessel.get_material_dataframe()
        #set the total amount of each material in ending vessels
        for name,data in mat_df.iterrows():
            if name in reactants:
                end_react[reactants[name]]+=data.Amount
            else:
                end_total[name] = end_total.get(name,0)+data.Amount
    #Making sure all of the materials still exist at the end
    for mat in start_total:
        if not mat in end_total:
            return False
    #perform the checksum
    for mat,amount in end_total.items():
        if not mat in start_total:
            return False
        if abs(amount-start_total[mat])>TOL:
            print(amount,start_total[mat])
            return False
    diff = get_diff(reaction.conc_coeff_arr,end_react-start_react)
        
    return np.dot(diff,diff)<TOL