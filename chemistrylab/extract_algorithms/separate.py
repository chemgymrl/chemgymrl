"""

"""
import numpy as np
from math import ceil
import numba



# Maximum varience of Gaussian peaks


# Array for x/height positions
x = np.linspace(0, 1, 100, endpoint=True, dtype=np.float32)


#from numba.pycc import CC
#cc = CC('separate_cc')

@numba.jit(cache=True,nopython=True)
#@cc.export('map_to_state', '(f4[:], f4[:],f4[:],f4[:],f4[:])')
def map_to_state(A, B, C, colors, x=x):
    """
    Uses the position and variance of each solvent to stochastically create a layer-view of the vessel
    
    Args:
        A (np.ndarray): The volume of each solvent
        B (np.ndarray): The current positions of the solvent layers in the vessel
        C (float): The current variance of the solvent layers in the vessel
        colors (np.ndarray): The color of each solvent

    Returns:
        Tuple[np.ndarray]: 
            - The solvent at each layer position (0.65 for air)
            - The index of the solvent at each position (len(B)-1 for air)

    Algorithm:

    1. Discretize the vessel into 100 layers each with one unit of volume
    2. Quantize the volumes into units of size sum(v)/100. (Round up agressively)
    3. Do a checksum to make sure these quantized volumes sum to 100
        i. If the sum of everything that isn't air is over 100, then decrease the solvent with the largest number of units
        ii. Otherwise you can just set the number of air units to 100-sum([all vi which aren't air])
    4. Find the position of the top layer
    5. For each of the quantized layers, gather the height of each gaussian at that layer position and sample a solvent proportional to this height
        i. This is approximately the same as doing an integral of the solvents distribution over the layer
        ii. Unfortunately, the solvent distributions don't add up to 1 so you have to normalize.
        iii. The distributions are more ballparks so you have to keep track of how many units you placed, and set the probability of the layer having a solvent to zero if all the units have already been placed
        iv. This also means you may not have placed all of your units by the time you are way outside the variance of your gaussian, so you should keep track of the lowest layer that still has units to place, and make sure those units are all placed once you start to go way past it.
    """
    # Create a copy of B for temporary changes
    B1 = np.copy(B)
    
    x = x*np.sum(A)
    #grab the index of the least dense material
    j_max = np.argmax(B)
    # If there are duplicates of the least dense material jmax won't work
    if np.sum(np.abs(B[j_max]-B)<1e-4)>1:
        j_max=-10

    # Array for layers  (1.)
    L = np.zeros(100, dtype=np.float32) + colors[-1]
    L2 = np.zeros(100, dtype=np.int32)+(len(colors)-1)

    # Initialize time variable such that Gaussians have normalized area
    C=np.clip(C,1e-10,None)

    #Note: MINP is linked to MINVAR
    #If you want to change one you have to change both
    #https://www.desmos.com/calculator/kzydc5ra3q

    width = C*3.5/2
    #MINVAR is 3.5 here
    MINP=0.21626516683

    
    sum_A = 1.0 if np.sum(A) == 0 else np.sum(A)
    # Number of pixels available for each solvent (2.)
    n=np.zeros(A.shape,dtype=np.int32)
    n[:]=((A / sum_A) * L.shape[0] + np.float32(0.95))
    # Since this agressively rounds up, fix any rounding issues
    count = np.sum(n[:-1]) - L.shape[0]
    if count>0:
        # 3.i
        for i in range(count):
            n[n.argmax()] -= 1
    else:
        # Make any unused pixels air (3.ii)
        n[-1] = -count
    
    # Loop over each layer pixel
    for l in range(L.shape[0]):
        # Map layer pixel position to x position
        k = l

        # 5.i
        P_raw = np.exp(-0.5 * (((x[k] - B) / C) ** 2))
        # MINP is a cutoff value to set the gaussian to 0
        P_clip = P_raw* ((P_raw > MINP) | (x[k]>B))#+ P_raw*(P_raw < MINP))
        
        # Calculate Gaussian values at current x position
        P = A /C * P_clip
        
        # Check to see if any phases have no pixels remaining
        past_due = False
        for j in range(P.shape[0]):
            if n[j] == 0:
                # Set Gaussian value to avoid placement by random set
                P[j] = 0.0
                P_raw[j]=0.0
                # Set Gaussian center extremely positive outside of range to avoid placement by default set
                B1[j] += 1e9

        # j_min is the index of the lowest gaussian which still has pixels to place (5.iv)
        j_min = np.argmin(B1-width)

        # Only need to place the least dense material
        if np.argmin(B1)==j_max and n[j_max]>0:
            L[l:] = colors[j_max]
            L2[l:]=j_max
            return L,L2
        # Sum of all Gaussians at this x position
        

        # Random number for mixing of layers
        r = np.random.rand()
        place_jmin = False
        #Below if part 5.iv
        # The current point has to be far to the RIGHT of the gaussian for it to have passed over
        if P_clip[j_min] < MINP and n[j_min]>0 and x[k]>B[j_min]:
            if P_clip[j_min]<1e-12:
                place_jmin=True
            else:
                P[j_min] = (A[j_min] / C[j_min]) * MINP
                #More likely to place j_min pixels the lower it's propability is
                #choice_ratio = 0.1*MINP**2/P_clip[j_min]
                #r2=np.random.rand()
                #if choice_ratio>=r2:
                #    place_jmin=True

        Psum = np.sum(P)

        # If x position is outside every Gaussian peak (5.iv and 5.iii)
        if Psum < 1e-6  or place_jmin:
            # Calculate the index of the most negative phase
            j = j_min
            # Set pixel value
            L[l] = colors[j]
            L2[l]=j
            # Subtract pixel for that phase
            n[j] -= 1
        # If x position is inside at least one Gaussian peak (5.i - 5.iii)
        else:
            p = 0.0
            # Loop until pixel is set
            for j in range(P.shape[0]):
                p += P[j]
                # If random number is less than relative probability for that phase (5.ii)
                if r - p / Psum < 1e-6:
                    # Set pixel value
                    L[l] = colors[j]
                    L2[l]=j
                    # Subtract pixel for that phase
                    n[j] -= 1
                    # End loop
                    break
            else: # This code runs when the loop didn't actually set any pixels and fills it with air
                L[l] = colors[j]
                L2[l]=j
                n[j] -= 1
    
    return L,L2


@numba.jit(cache=True,nopython=True)
#@cc.export('mix', '(f4[:], f4[:], f4[:], f4[:], f4[:], f4, f4[:], f4[:], f4[:], f4[:,:],f4)')
def mix(v, Vprev, v_solute, B, C, C0 , D, Spol, Lpol, S, mixing):
    """
    Calculates the positions and variances of solvent layers in a vessel, as well as the new solute amounts, based on the given inputs.

    Args:
        v (np.ndarray): The volume of each solvent
        Vprev (np.ndarray): The volume of each solvent on the previous iteration
        v_solute (np.ndarray): The specific volume of each solute (litres per mol)
        B (np.ndarray): The current positions of the solvent layers in the vessel
        C (np.ndarray): The current variances of the solvent layers in the vessel
        C0 (float) The current variance of solutes in the vessel
        D (np.ndarray): The density of each solvent
        Spol (np.ndarray): The relative polarities of the solutes
        Lpol (np.ndarray): The relative polarities of the solvents
        S (np.ndarray): The current amounts of solutes in each solvent layer (2D array)
        mixing (float): The time value assigned to a fully mixed solution

    Returns:
        Tuple[np.ndarray]:
            - layers_position: An array of floats representing the new positions of the solvent layers in the vessel
            - layers_variance: An array of floats representing the new variances of the solvent layers in the vessel
            - new_solute_amount: An array of floats representing the new amounts of solutes in each solvent layer
            - var_layer: Modified layer variances which account for the extra volume due to dissolved solutes

    Algorithm (Solvent):

    1. Using the volumes and densities of each solvent, determine where each solvent's center of mass should be at t-> inf
    2. Determine the speed in which each solvent should separate out using the densities
    3. Handle any external changes to the solving (pouring in/out) using v and Vprev
        i. Since there is an injective map between variance and time, it is easier to work with variance
            a) Initial variance is sum(v)/sqrt(12) [gaussian approximation of a uniform distribution]
            b) Final variance is vi/MINVAR -> MINVAR should probably be around sqrt(12) still (but be <=)
        ii. Pouring in a solvent should kind of mix around the solution, and since the max variance is sum(v)/sqrt(12) adding in dv/sqrt(12) seems reasonable
        iii. For the solvent actually being added, we can assume you are pouring into the top, so it should be mixed the closer to the bottom the solvent layer is. It should also be mixed more depending on how much you are adding.
        iv. If adding a solvent causes things to be mixed around a bunch, it should end up mixing the solutes too
    4. Get a time-like variable saying much each solvent is settled using the current variance (Recall the map is injective)
    5. Increment this by the mixing parameter
        i. If time is being decreased by the mixing parameter, we first set T<= Tmax so something which settled for a long time still mixes reasonably fast (and also as T->inf the map between variance and time gets sus cuz of floats)
    6. Use this incremented time to update your layer positions, as well as layer variances

    Algorithm (Solute):
    TODO: Write this out

    """
    
    s=C*1
    x=B*1
    # CONSTANTS
    MINVAR=np.float32(3.5)
    SCALING=np.float32(1e-2)
    t_scale = 25
    #Cmix = np.float32(2.0)
    tmix = np.float32(-1.6120857137646178)#-1.0 * np.log(Cmix * np.sqrt(2.0 * np.pi))
    tseparate = np.float32(-1.47)
    TOL=np.float32(1e-12)
    E3 = np.float32(1e-3)
    MAXVAR = np.float32(3.0)

    # copy mixing for the solutes in case you need to modify it
    solute_mixing = 0
    #figure out where the gaussians should end up at T-> inf
    order=np.argsort(D)[::-1]
    Vtot= np.sum(v) #Total volume



    #Get convergence speeds based off of how different the densities are
    diff = np.zeros(D.shape[0],dtype=np.float32)
    for i in range(diff.shape[0]):
        for j in range(0, i):
            diff[j] -= (D[j] - D[i])
        for j in range(i+1, D.shape[0]):
            diff[j] -= (D[j] - D[i])
    diff = np.clip(np.abs(diff),E3,None)

    max_var = Vtot/MAXVAR
    solvent_mixing=0
    #adjust variance
    for i in range(Lpol.shape[0]):
        # Figure out how much the volume has changed
        dv = v[i] - Vprev[i]
        # Make sure variance is at least as big as fully separated variance
        cur_var= max(s[i],v[i]/MINVAR)
        # Add some variance to each solvent
        #s+=dv/MAXVAR
        # Extra mixing dependant on the position and how much was added
        if dv>1e-6:
            new_var = (dv/(np.abs(v[i]-dv)+TOL))*((Vtot-x[i])/MAXVAR)
            new_var = min(max_var, max(cur_var,new_var))
            #mix surrounding solvents (excludes air)
            if i<v.shape[0]-1:s[:v.shape[0]-1]+= dv/MAXVAR+(new_var-cur_var)/2
            #Mix solvent
            s[i]=new_var
            #TODO: Set extra mixing of solutes
            var_ratio = (new_var-cur_var)/(abs(max_var-cur_var)+TOL)
            solute_mixing = min(solute_mixing, (tmix-tseparate)*var_ratio )
            
        else:
            s[i] = min(max_var, s[i]+dv/MINVAR) 
            
    if (s.shape[0]!=v.shape[0]):
        s0,s=s,v/MINVAR
        s[:Lpol.shape[0]]=s0[:Lpol.shape[0]]

    

    #Get the mixing-time variable
    sf = v/MINVAR # final variances
    si = Vtot/MAXVAR # initial variances
    s=np.clip(s,sf+TOL,si-TOL)
    #ratio -> inf as t -> inf
    ratio = np.clip((si-sf)/(s-sf),1,None)

    #Elapsed time T is [0,inf) and increases monotonely with ratio
    T = np.sqrt(np.log(ratio)/2)*Vtot
    #Set T in line with air for non-solvents
    T[Lpol.shape[0]:] = T[-1]/diff[-1]*diff[Lpol.shape[0]:]

    # Add any extra time
    ratio = np.clip(v/Vtot,0,1-E3)
    dt = mixing*diff/(1-ratio)**2*SCALING
    # Do a cap on T when mixing since you should always be able to stir the vessel
    # Even if the vessel has been settling for 100 years
    if mixing< -1e-4:
        T=np.clip(T,0,3.278)
        T_max = np.float32(3.278)*dt/dt.min()
        T = (T>T_max)*T_max+(T<=T_max)*T

    T+=dt
    #Make sure Time is >= 0 (0 is fully mixed time)
    T=np.clip(T,0,None)


    #Update variance
    g = np.exp(-2*(T/Vtot)**2)
    C = sf+(si-sf)*g

    #Math for position vs time:
    #https://www.desmos.com/calculator/imukub1xzr 

    # A are the volumes used for dissolving
    A=v[:Lpol.shape[0]]

##############################[Mixing / Separating Solutes]#######################################

    t = np.float32(-np.log(C0 * np.sqrt(2.0 * np.pi)) )

    # Mixing should always mix at least a bit   
    if mixing<0 or solute_mixing<0:
        t=min(t,tseparate)

    mixing+=solute_mixing
    # Check if fully mixed already
    if t + mixing < tmix:
        mixing = tmix - t
    t += mixing


    Scur = np.copy(S)
    # only do the calculation if there are two or more solvents
    if not ((len(A) < 2) or A.sum()-A.max()<1e-12 or abs(mixing)<1e-12):
    # Update amount of solute i in each solvent
      for i in range(S.shape[0]):

        Ssum = np.sum(Scur[i])
        if Ssum<1e-6:
            continue    

        # Calculate the relative and weighted polarity terms
        Ldif = np.float32(0)
        Ldif0 = np.float32(0)
        for j in range(Lpol.shape[0]):
            Ldif += np.abs(Spol[i] - Lpol[j])
            Ldif0 += A[j] * np.abs(Spol[i] - Lpol[j])

        # Calculate total amount of solute
        
        
        # Note A*re_weight has the same sum as A
        re_weight = (1 - (np.abs(Spol[i] - Lpol) / Ldif)) / (1 - (Ldif0) / (np.sum(A) * Ldif))
        #coeff for current timestep
        alpha = np.exp(t_scale*(tmix - t))
        #coeff for prev timestep
        beta = np.exp(t_scale*(tmix - (t - mixing)))
        # Calculate the ideal amount of solute i in each solvent for the current time step
        St = (Ssum * A / np.sum(A)) * (alpha + (1-alpha)*re_weight)
        # Calculate the ideal amount of solute i in each solvent for the previous time step
        St0 = (Ssum * A / np.sum(A)) * (beta + (1 - beta) * re_weight)
        #print(S[i],St0,St,re_weight,mixing,alpha,beta)
        if np.abs(t - mixing - tmix) > 1e-9:
            #Square of the cosine between the two distributions
            cosine = (St0*S[i]).sum()/((S[i]**2).sum()*(St**2).sum())**0.5
            # Moving backwards in time
            if mixing<0:
                #Make sure mixing always happens reasonably well
                cosine = max(cosine,1-cosine)
                S[i] = S[i]*(1-cosine) + St*cosine
            # Moving Forwards in time
            else:
                #scale cosine to be higher if you wait longer (bad approximation)
                cosine = max(min(1,cosine**1.5*(mixing*t_scale*5)**1.5),1e-2)
                # Move towards projected step St with cosine step size
                S[i] = S[i]*(1-cosine) + St*cosine
        else:
            S[i] = St

    C0 = np.float32( np.exp(-1.0 * t) / np.sqrt(2.0 * np.pi) )

    #Update layer volumes to include their dissolved solutes
    v_layer = v.copy()
    for i in range(S.shape[0]):
        for j in range(Lpol.shape[0]):
            vol = S[i][j]*v_solute[i]
            # Add solute volume to solvent volume
            v_layer[j]+=vol
            #reduce amount of air
            v_layer[-1]-=vol

    # Calculate final layer positions
    means = np.zeros(D.shape[0],dtype=np.float32)
    Vtot = np.float32(0)
    for i in order:
        means[i] = Vtot+v_layer[i]/2
        Vtot+=v_layer[i]

    #update positions
    B = means+(Vtot/2-means)*g
    # Adjust variances for dissolved volumes
    sf2 = v_layer/MINVAR 
    var_layer = sf2+(si-sf2)*g

    return B, v_layer, C,C0, S, var_layer






if __name__ == "__main__":
    
    cc.compile()