import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from chemistrylab.reactions.available_reactions.fict_react2 import PRODUCTS as FRtargs
from chemistrylab.reactions.available_reactions.fict_react2 import REACTANTS as FRchoices
from chemistrylab.reactions.available_reactions.chloro_wurtz import PRODUCTS as CWtargs
from chemistrylab.reactions.available_reactions.chloro_wurtz import REACTANTS as CWchoices
from RadarGraph import *

import numba

###################################################Loading Data########################################################
calc_return = default_obj = lambda x: x.Reward.sum()/x.Done.sum()
worst_obj = lambda x: -x.Reward.sum()/x.Done.sum()


def load_rollouts(env: str,obj = default_obj,last: bool = True,verbose:bool=False,TOL:float = 1e-4):
    """Retrieve RL rollouts of each algorithm from the file system
    
    Parameters:
        obj (function) - Method to measure how good the rollout is
        env (str) - The environment you want rollouts from
        last (bool) - Use the rollout of the last timestep if true and the best performing timestep if false
        verbose (bool) - Set to true if you want filepaths and objective evaluations printed
        TOL (int)      - Tolerance for evaluating two runs as the same objective-wise (in which case the run with more
                         episodes is preferred)
    """
    #nt is for windows
    delim = "\\" if os.name=='nt' else "/"
    
    folder = f"MODELS{delim}{env}" if len(env.split(delim))==1 else env
    algoidx = [i for i,string in enumerate(folder.split(delim)) if "-v" in string][0]
    data = dict()
    rollName=["best_rollout","rollout"][last]
    for a,b,c in os.walk(folder):
        if rollName in c:
            algo = a.split(delim)[algoidx+1]
            df1 = pd.read_pickle(a+delim+rollName)
            if verbose:print(a,"|",0 if obj is None else obj(df1))
            if not algo in data:
                data[algo]=df1
            else:
                df0=data[algo]
                if obj is None:
                    data[algo]=pd.concat([df0,df1],ignore_index=True)  
                #prefer rollouts with more episodes when objective ranking is similar
                elif abs(obj(df1)-obj(df0))<TOL:
                    if verbose:print(df1.Done.sum(),df0.Done.sum(),df0.Done.shape)
                    if df1.Done.sum()>df0.Done.sum():
                        data[algo]=df1
                #prefer higher ranking ones
                elif obj(df1)>obj(df0):
                    data[algo]=df1
    return data



@numba.jit
def fast_compare(result,r,l,steps):
    l_sum=0
    for i in range(r.shape[0]): 
        idx=int(l_sum/steps)
        if idx>=result.shape[0]:
            return
        result[idx][0]+=r[i]
        result[idx][1]+=1
        l_sum+=l[i]
        
        
def merge_varying_graphs(folder = "./RL_rollout/MODELS/FictReact-v2/TD3",steps:int=100,verbose:bool=False,separate_runs=True):
    """Collect results from multiple runs and merge them into one run
    
    Inputs:
        folder - Directory in which to look for monitor.csv files
        steps - size of the step window. For fixed length environments set this to a multiple of the episode length
        verbose - Whether or not to print additional details
    Outputs:
        returns (Array) - Shape [runs,timesteps/steps] array of average return vs step for each run
        episode_counts (Array) - [runs,timesteps/steps] array of episode counts for each step window and run    
    """
    tot=0
    best=-1e10
    bestdir=""
    RESULT=np.zeros([1,2])
    all_results=[]
    for a,b,c in os.walk(folder):
        if len(c)>0:
            
            if "rollout" in c:
                rollout = pd.read_pickle(a+"/rollout")
                rew = rollout[rollout.Done==True].Reward.mean()
                if rew>best:
                    best = rew
                    bestdir=a
            
            paths = [a+"/"+fn for fn in c if "monitor" in fn]
            if len(all_results)>0:
                RESULT=np.zeros(all_results[0].shape)
            for p in paths:
                if tot==0:
                    frame = pd.read_csv(p,header=1)
                    if verbose:print(frame.l.sum())
                    RESULT=np.zeros([frame.l.sum()//steps,2])
                if separate_runs:
                    frame = pd.read_csv(p,header=1)
                    fast_compare(RESULT,np.stack(frame.r),np.stack(frame.l),steps)
                else:
                    frame = pd.read_csv(p,header=1)
                    if tot>0:RESULT=np.zeros(all_results[0].shape)
                    fast_compare(RESULT,np.stack(frame.r),np.stack(frame.l),steps)
                    all_results+=[RESULT]
                    
                tot+=1
            if len(paths)>0 and separate_runs:all_results+=[RESULT]
    if verbose:print(tot,[bestdir,best])
    #print(*[a.shape for a in all_results])
    all_results=np.stack(all_results)
    #If there are zero instances then divide by 1 and plot 0
    returns=all_results[:,:,0]/np.clip(all_results[:,:,1],1,1e10)
    episode_counts=all_results[:,:,1]
    return returns,episode_counts

#################################################################################################################


###########################################Parsing Data##########################################################

def target_subset(frame,N,i):
    """
    Filters the rollout for episodes which have a specific target
    Inputs:
        Frame (dataframe) - Pandas Dataframe containing gym information
        N (int)           - Number of targets in your environment
        i (int)           - The index of your target as it appears in the observation space
        
    Outputs:
        cframe (dataframe) - Subset of your Pandas dataframe with only episodes of target i
    
    """
    obs = np.stack(frame.InState)
    cframe=frame[obs[:,-N+i]>0.9]
    return cframe



def actions_by_time(frame):
    """Gives the mean action at each timestep of your rollout dataframe"""
    min_t,max_t = frame.Step.min(),frame.Step.max()
    mean_act=[]
    for t in range(min_t,max_t+1):
        mean_act+=[frame.Action[frame.Step==t].mean()]
    return np.array(mean_act)








def get_conditional_rewards(frame,targets=CWtargs):
    """
    Gives returns conditioned on the different targets
    Inputs:
        Frame (dataframe) - Pandas Dataframe containing gym information
        targets (list) - List of N targets (reaction products)
        
    Outputs:
        targets
        rew (List<float>) - List of size N containing the average return given each target
    
    """
    # turn observation column into a numpy array
    obs = np.stack(frame.InState)
    N=len(targets)
    rew=[]
    for i in range(N):
        #gather all data where the target is targets[N]
        cframe=frame[obs[:,-N+i]>0.9]
        #Obtain the mean reward of these episodes
        rew+=[calc_return(cframe)]
    return [targets,np.array(rew)]

def get_conditional_actions(frame,targets=CWtargs):
    """
    Gives actions conditioned on the different targets, meant for continuous action spaces
    Inputs:
        Frame (dataframe) - Pandas Dataframe containing gym information
        targets (list) - List of N targets (reaction products)
        
    Outputs:
        targets
        act (List<array>) - List of size N containing the mean action given each target
    
    """
    # turn observation column into a numpy array
    obs = np.stack(frame.InState)
    N=len(targets)
    act=[]
    for i in range(N):
        #gather all data where the target is targets[N]
        cframe=frame[obs[:,-N+i]>0.9]
        #Obtain the mean action of these episodes
        act+=[cframe.Action.mean()]
    return [targets,act]




def get_discrete_actions(frame,N=None,N2=None):
    """
    Gives distribution of actions (index 0) taken as well as the average value of the actions at index 1
    Inputs:
        Frame (dataframe) - Pandas Dataframe containing gym information        
    Outputs:
        act0 (list<float>) - Action (index 0) distribution
        act1 (list(float)) - Average action at index 1
    
    """
    # turn observation column into a numpy array
    act = np.stack(frame.Action)    
    if len(act.shape)<2:
        act0=act
        act=np.zeros(act0.shape+(2,),dtype=np.int32)
        act[:,0]=act0//N2
        act[:,1] = act0%N2
        
    if N is None:
        N = np.max(act[:,0])
    N0= np.max(act[:,1])
    act0=[]
    act1=[]
    #print(N)
    for i in range(N+1):
        #gather all data where the target is targets[N]
        cframe=act[act[:,0]==i]
        
        #print(cframe)
        #Obtain the mean action of these episodes
        act0+=[len(cframe)/act.shape[0]]
        if len(cframe)==0:
            act1+=[0]
        else:
            act1+=[cframe[:,1].mean()/N0]
    return [act0,act1]




##################################################################################################



#####################################Plotting Data#################################################

def stat_show(radar_info:dict,spoke_labels:list,labels,figsize=(22,7),gridlines=[0.0,0.4,0.8,1.2,1.6,2.0],relative=True):
    """Radar Plots a set of stats
    
    Inputs:
        radar_info (dict) - A dictionary containing the data for all the radar graphs, it should be formatted like so:
                        {graph name: graph data} where graph data is a 2D array of shape [len(labels),len(spoke_labels)]
                        
        spoke_labels (list) - A list of text labels for each spoke (vertex) of the radar graph
        
        labels (list) - A list of text labels telling you what each shaded area means (Ex the first shaded area is Return)
        
        figsize - How large to make the figure
        
        relative - Whether or not to scale the last graph (Are the initial graphs 'relative' to the final graph?)
    
        gridlines (list) - list of where to put each gridline
    """
    
    colors = ["r","g","b","c","y","m","k"]
    info = ([spoke_labels]+[(key,radar_info[key]) for key in radar_info])
    
    
    theta = radar_factory(len(info[0]), frame='polygon')
    
    fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=len(radar_info),subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.5, hspace=0.25, top=0.85, bottom=0.05)
    
    make_radar(theta,axs,info,colors = "r",gridlines=gridlines)
    legend = axs[0].legend(labels, loc=(0.9, .95),labelspacing=0.1, fontsize='small')
    

    if relative:
        #scale all but the heurstic the same
        for ax in axs.flat[:-1]:
            ax.set_rmin(max(0,min([a[0].min() for (c,a) in info[1:-1]]+[0])))
            ax.set_rmax(max([a[0].max() for (c,a) in info[1:-1]]+[1]))
    else:
        #scale them all the same
        for ax in axs.flat:
            ax.set_rmin(min([a[0].min() for (c,a) in info[1:]]+[0]))
            ax.set_rmax(max([a[0].max() for (c,a) in info[1:]]))
    
    return fig,axs