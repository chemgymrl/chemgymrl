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

    width = C
    
    #adjust variance 
    C = C*2/3.5
    #MINP is the probability at which the left side of the gaussian is clipped
    MINP=0.21626516683

    
    sum_A = 1.0 if np.sum(A) == 0 else np.sum(A)
    # Number of pixels available for each solvent (2.)
    n=np.zeros(A.shape,dtype=np.int32)
    r = (A / sum_A) * L.shape[0]
    n[:]= r
    # Since this agressively rounds up, fix any rounding issues
    count = np.sum(n) - L.shape[0]
    if count>=0:
        # 3.i
        for i in range(count):
            n[n.argmax()] -= 1
    else:
        r -= n
        #CONSIDER: Reduce the probability of inceasing air with: 
        r[-1]=1e-12
        for i in range(-count):
            # get a random choice without replacement
            cum_dist = np.cumsum(r)
            cum_dist /= cum_dist[-1]
            rnd = np.random.rand()
            index = np.searchsorted(cum_dist, rnd, side="right")
            n[index] += 1
            r[index]=0
    
    # Loop over each layer pixel
    for l in range(L.shape[0]):
        # 5.i
        P_raw = np.exp(-0.5 * (((x[l] - B) / C) ** 2))
        # Gaussian can only be found to the right of the width parameter
        P_clip = P_raw*(x[l]>=B-width)
        # Calculate Gaussian values at current x position
        P = A /C * P_clip
        # Check to see if any phases have no pixels remaining
        for j in range(P.shape[0]):
            if n[j] == 0:
                # Set Gaussian value to avoid placement by random set
                P[j] = 0.0
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
        if n[j_min]>0 and x[l]>B[j_min]+width[j_min]:
            if P_clip[j_min]<1e-12:
                place_jmin=True
            else:
                P[j_min] = (A[j_min] / C[j_min]) * MINP

        P_cum = np.cumsum(P)
        # If x position is outside every Gaussian peak (5.iv and 5.iii)
        if P_cum[-1] < 1e-6  or place_jmin:
            # Calculate the index of the most negative phase
            j = j_min
            # Set pixel value
            L[l] = colors[j]
            L2[l]=j
            # Subtract pixel for that phase
            n[j] -= 1
        # If x position is inside at least one Gaussian peak (5.i - 5.iii)
        else:
            #normalize
            P_cum/=P_cum[-1]
            # Loop until pixel is set
            for j in range(P.shape[0]):
                # If random number is less than relative probability for that phase (5.ii)
                if r - P_cum[j] < 1e-6:
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
def mix(v, Vprev, v_solute, B, T0, C0 , D, Spol, Lpol, S, mixing):
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
            - layers_position: 1D array with the new positions of the solvent layers in the vessel
            - layers_volume: The volume of each layer adjusted to how much solute is dissolved in it
            - layers_settle_time: 1D array representing how far along settling is for each layer
            - variance: Variance measure for the solute separation
            - new_solute_amount: 1D array with the new amounts of solutes in each solvent layer
            - var_layer: Modified layer variances which account for the extra volume due to dissolved solutes

    Algorithm:

    1. Calculate change in time (run time backwards for mixing, and forwards for settling, T=0 is fully mixed)
    2. Determine equilibrium values for dissolved amounts using polarity
    3. Scale solute amounts between fully mixed and equilibrium using the time component
    4. Use the new dissolved amounts to adjust layer volumes
    5. Use the time variable to determine how much the layers have pushed eachother apart.

    """
    
    x=B*1
    SCALING=np.float32(2e-1)
    t_scale = 125
    tseparate = np.float32(0.142)
    TOL=np.float32(1e-12)
    E3 = np.float32(1e-3)
    # copy mixing for the solutes in case you need to modify it
    solute_mixing = 0
    Vtot= np.sum(v) #Total volume

    # Approximate separation time
    tau = 2/(D.max()-D.min()+E3)
    T = T0

    for i in range(Lpol.shape[0]):
        # Figure out how much the volume has changed
        dv = v[i] - Vprev[i]
        # Adding in solvents reverses the separation process
        if dv>1e-6:            
            ratio = (dv/(np.abs(v[i]-dv)+TOL))*(Vtot-x[i])
            #mix solvents
            if i<v.shape[0]-1:
                T -= ratio*tau/4     
            #TODO: Set extra mixing of solutes
            solute_mixing = min(solute_mixing, -tseparate*ratio )

    # Do a cap on T when mixing  (set max settling time)
    if mixing< -1e-4 or solute_mixing<-1e-4:
        T=min(T,tau)

    dt = mixing*SCALING
    T+=dt
    #Make sure Time is >= 0 (0 is fully mixed time)
    T=max(T,0)

    # A are the volumes used for dissolving
    A=v[:Lpol.shape[0]]

##############################[Mixing / Separating Solutes]#######################################

    Scur = np.copy(S)
    # only do the calculation if there are two or more solvents
    if not ((len(A) < 2) or A.sum()-A.max()<1e-12 or abs(T-T0)<1e-12):
        a = A/A.sum()
        # Update amount of solute i in each solvent
        for i in range(S.shape[0]):
            Ssum = np.sum(Scur[i])
            if Ssum<1e-6:
                continue    
            polarity_diff = np.abs(Spol[i] - Lpol)
            # Re weight based off of polarity difference and solvent amount
            #re_weight = a*(polarity_diff.sum()-polarity_diff)
            re_weight = a/(polarity_diff+1e-2)

            re_weight /= re_weight.sum()
            #coeff for current timestep
            alpha = np.exp(-t_scale*T)
            #coeff for prev timestep
            beta = np.exp(-t_scale*T0)
            # Calculate the ideal amount of solute i in each solvent for the current time step
            St = Ssum *(alpha*a + (1-alpha)*re_weight)
            # Calculate the ideal amount of solute i in each solvent for the previous time step
            St0 = Ssum * (beta*a + (1 - beta) * re_weight)

            if np.abs(T-T0) > 1e-9:
                #Square of the cosine between the two distributions
                cosine = (St0*S[i]).sum()/((S[i]**2).sum()*(St**2).sum())**0.5
                # Moving backwards in time
                if T-T0<0:
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

    C0 = T*5

    ########################### MOVING LAYERS ##################################


    #Update layer volumes to include their dissolved solutes
    v_layer = v.copy()
    for i in range(S.shape[0]):
        for j in range(Lpol.shape[0]):
            vol = S[i][j]*v_solute[i]
            # Add solute volume to solvent volume
            v_layer[j]+=vol
            #reduce amount of air
            v_layer[-1]-=vol

    #final layer positions
    means = np.zeros(D.shape[0],dtype=np.float32)
    #final layer variances
    var_layer = np.zeros(D.shape[0],dtype=np.float32)
    # Total volume
    Vtot=np.sum(v_layer)

    # Math for position vs time:
    # https://www.desmos.com/calculator/zdxelweyw4 

    #Roughly simulates how layers will push eachother apart
    for i in range(D.shape[0]):
        di = D[i]
        # NOTE: You can swap tanh(x) with np.clip(x,-1,1) but you need to adjust tau and SCALING
        #total separation can be thought of as a sum of pair-wise separations
        means[i] = Vtot/2 - np.sum(np.tanh((di-D)*T)*v_layer)/2
        # same goes for final variances. . .
        var_layer[i] = Vtot/2 - np.sum(np.tanh(np.abs(di-D)*T)*v_layer)/2

    return means, v_layer, T , C0, S, var_layer



if __name__ == "__main__":
    
    cc.compile()