"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from math import ceil
from copy import deepcopy

# Maximum varience of Gaussian peaks
Cmix = 2.0

# Array for x/height positions
x = np.linspace(0, 1, 1000, endpoint=True, dtype=np.float32)

import numba

@numba.jit(cache=True)
# Function to map the separation Gaussians to discretized state
def map_to_state(A, B, C, colors, x=x):
    """
    Uses the position and variance of each solvent to stochastically create a layer-view of the vessel

    Args:
    - A (np.ndarray): The volume of each solvent
    - B (np.ndarray): The current positions of the solvent layers in the vessel
    - C (float): The current variance of the solvent layers in the vessel
    - colors (np.ndarray): The color of each solvent

    Returns:
    - L (np.ndarray): The solvent at each layer position (0.65 for air)
    - L2 (np.ndarray): The index of the solvent at each position (len(B)-1 for air)
    """
    # Create a copy of B for temporary changes
    B1 = np.copy(B)
    
    x = x*np.sum(A)
    #grab the index of the least dense material
    j_max = np.argmax(B)
    # Array for layers at each time step
    L = np.zeros(100, dtype=np.float32) + colors[-1]
    L2 = np.zeros(100, dtype=np.int32)+(len(colors)-1)

    # Initialize time variable such that Gaussians have normalized area
    C=np.clip(C,1e-10,None)

    #Note: MINP is linked to MINVAR
    #If you want to change one you have to change both
    #https://www.desmos.com/calculator/jc0ensg38o
    MINP=0.13533

    # Number of pixels available for each phase
    sum_A = 1.0 if np.sum(A) == 0 else np.sum(A)
    n = ((A / sum_A) * L.shape[0])
    count = 0
    for i in np.where((n > 1e-8) | (n < 0.5 - 1e-8))[0]:
        n[i] = ceil(n[i])
        count += 1

    #n = n.astype(int)
    n0=n
    n=np.zeros(n0.shape,dtype=np.int32)
    n[:]=n0
    if np.sum(n[:-1]) > L.shape[0]:
        for i in range(count):
            n[n.argmax()] -= 1
    else:
        # Take rounding errors into account
        n[-1] = 0
        n[-1] = L.shape[0] - np.sum(n)
    
    # Loop over each layer pixel
    for l in range(L.shape[0]):
        # Map layer pixel position to x position
        k = int(((l + 0.5) / L.shape[0]) * x.shape[0])
        #print(x[k],end='|')

        P_raw = np.exp(-0.5 * (((x[k] - B) / C) ** 2))
        # MINP is a cutoff value to set the gaussian to 0
        P_raw = P_raw* ((P_raw > MINP) + P_raw*(P_raw < MINP))
        
        # Calculate Gaussian values at current x position
        P = A /C * P_raw
        

        # Check to see if any phases have no pixels remaining
        past_due = False
        for j in range(P.shape[0]):
            if n[j] == 0:
                # Set Gaussian value to avoid placement by random set
                P[j] = 0.0

                # Set Gaussian center extremely positive outside of range to avoid placement by default set
                B1[j] += 1e9

        # j_min is the index of the lowest gaussian which still has pixels to place
        j_min = np.argmin(B1)

        # Only need to place the least dense material
        if j_min==j_max:
            L[l:] = colors[j_max]
            L2[l:]=j_max
            return L,L2
        # Sum of all Gaussians at this x position
        Psum = np.sum(P)

        # Random number for mixing of layers
        r = np.random.rand()
        place_jmin = False
        if P_raw[j_min] < MINP:
            if P_raw[j_min]<1e-12:
                place_jmin=True
            else:
                #More likely to place j_min pixels the lower it's propability is
                choice_ratio = 0.1*MINP**2/P_raw[j_min]
                r2=np.random.rand()
                if choice_ratio>=r2:
                    place_jmin=True


        # If x position is outside every Gaussian peak (default set)
        if Psum < 1e-6  or place_jmin:
            # Calculate the index of the most negative phase
            j = j_min
            # Set pixel value
            L[l] = colors[j]
            L2[l]=j
            # Subtract pixel for that phase
            n[j] -= 1
        # If x position is inside at least one Gaussian peak (random set)
        else:
            p = 0.0
            # Loop until pixel is set
            for j in range(P.shape[0]):
                p += P[j]
                # If random number is less than relative probability for that phase
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


@numba.jit(cache=True)
def mix(v, Vprev, B, C, C0 , D, Spol, Lpol, S, mixing):
    """
    Calculates the positions and variances of solvent layers in a vessel, as well as the new solute amounts, based on the given inputs.

    Args:
    - v (np.ndarray): The volume of each solvent
    - B (np.ndarray): The current positions of the solvent layers in the vessel
    - C (float): The current variance of the solvent layers in the vessel
    - D (np.ndarray): The density of each solvent
    - Spol (np.ndarray): The relative polarities of the solutes
    - Lpol (np.ndarray): The relative polarities of the solvents
    - S (np.ndarray): The current amounts of solutes in each solvent layer (2D array)
    - mixing (float): The time value assigned to a fully mixed solution

    Returns:
    - layers_position (np.ndarray): An array of floats representing the new positions of the solvent layers in the vessel
    - layers_variance (np.ndarray): An array of floats representing the new variances of the solvent layers in the vessel
    - new_solute_amount (np.ndarray): An array of floats representing the new amounts of solutes in each solvent layer
    - Sts (np.ndarray): TODO: Figure out what this stores

    Note:
    - This function calculates the new positions and variances of the solvent layers using the layer separation equations described in the documentation.
    - The new solute amounts are calculated based on the relative polarities of the solutes and solvents using the solute dispersion equation described in the paper.
    """


    
    s=C*1.0
    x=B*1.0
    # CONSTANTS
    MINVAR=4.0
    SCALING=1e-2
    t_scale = 25
    tmix = -1.0 * np.log(Cmix * np.sqrt(2.0 * np.pi)) #-1.6120857137646178
    tseparate = -1.47


    # copy mixing for the solutes in case you need to modify it
    solute_mixing = 0
    #figure out where the gaussians should end up at T-> inf
    order=np.argsort(D)[::-1]
    Vtot=0 #Total volume
    means = np.zeros(D.shape[0])
    for i in order:
        means[i] = Vtot+v[i]/2
        Vtot+=v[i]


    #Get convergence speeds based off of how different the densities are
    diff = np.zeros(D.shape[0])
    for i in range(diff.shape[0]):
        for j in range(0, i):
            diff[j] -= (D[j] - D[i])
        for j in range(i+1, D.shape[0]):
            diff[j] -= (D[j] - D[i])
    diff = np.clip(np.abs(diff),1e-2,None)

    max_var = Vtot/3.46
    
    #adjust variance
    for i in range(v.shape[0]):
        # Figure out how much the volume has changed
        dv = v[i] - Vprev[i]
        # Make sure variance is at least as big as fully separated variance
        cur_var= max(s[i],v[i]/MINVAR)
        s+=dv/3.46
        # Extra mixing dependant on the position and how much was added
        if dv>1e-6:
            new_var = (dv/(np.abs(v[i]-dv)+1e-6))*((Vtot-x[i])/3.46)
            new_var = min(max_var, max(cur_var,new_var))
            s[i]=new_var
            #TODO: Set extra mixing of solutes
            var_ratio = (new_var-cur_var)/(abs(max_var-cur_var)+1e-6)
            solute_mixing = min(solute_mixing, (tmix-tseparate)*var_ratio )

    #Get the mixing-time variable
    sf = v/MINVAR # final variances
    si = Vtot/3.46 # initial variances
    s=np.clip(s,sf+1e-10,si-1e-10)
    #ratio -> inf as t -> inf
    ratio = np.clip((si-sf)/(s-sf),1,None)

    #Elapsed time T is [0,inf) and increases monotonely with ratio
    T = np.sqrt(np.log(ratio)/2)*Vtot
    # Add any extra time
    
    ratio = np.clip(v/Vtot,0,0.999)
    dt = mixing*diff/(1-ratio)**2*SCALING
    # Do a cap on T when mixing since you should always be able to stir the vessel
    # Even if the vessel has been settling for 100 years
    if mixing<-1e-4:
        T_max = 3.278*dt/dt.min()
        T = (T>T_max)*T_max+(T<=T_max)*T

    T+=dt
    #Make sure Time is >= 0 (0 is fully mixed time)
    T=np.clip(T,0,None)

    #update positions
    c=1.2/(np.abs(means-Vtot/2)+1e-6)
    c=np.clip(c,1e-3,30)
    f=np.log(1+(np.exp(c)-1)*np.exp(-c*T))/c
    B= means+(Vtot/2-means)*f
    #Update variance
    g = np.exp(-2*(T/Vtot)**2)
    C = sf+(si-sf)*g

    # A are the volumes used for dissolving
    A=v[:-1]

##############################[Mixing / Separating Solutes]#######################################

    t = -1.0 * np.log(C0 * np.sqrt(2.0 * np.pi))

    # Mixing should always mix at least a bit   
    if mixing<0 or solute_mixing<0:
        t=min(t,tseparate)

    mixing+=solute_mixing
    # Check if fully mixed already
    if t + mixing < tmix:
        mixing = tmix - t
    t += mixing


    Sts = np.zeros(S.shape)
    Scur = np.copy(S)
    # only do the calculation if there are two or more solvent
    if (len(A) < 2) or A.sum()-A.max()<1e-12 or abs(mixing)<1e-12:
        C0 = np.exp(-1.0 * t) / np.sqrt(2.0 * np.pi)
        return B, C, C0 , S, Sts

    
    # Update amount of solute i in each solvent
    for i in range(S.shape[0]):

        Ssum = np.sum(Scur[i])
        if Ssum<1e-6:
            continue    

        # Calculate the relative and weighted polarity terms
        Ldif = 0
        Ldif0 = 0
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

    C0 = np.exp(-1.0 * t) / np.sqrt(2.0 * np.pi)
    return B, C,C0, S, Sts

