import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#from chemistrylab.reactions.available_reactions.fict_react2 import PRODUCTS as FRtargs
#from chemistrylab.reactions.available_reactions.fict_react2 import REACTANTS as FRchoices
#from chemistrylab.reactions.available_reactions.chloro_wurtz import PRODUCTS as CWtargs
#from chemistrylab.reactions.available_reactions.chloro_wurtz import REACTANTS as CWchoices
FRtargs=[
    "fict_E",
    "fict_F",
    "fict_G",
    "fict_H",
    "fict_I"
 ]

FRchoices=[
    "fict_A",
    "fict_B",
    "fict_C",
    "fict_D",
]

CWtargs=[
    "dodecane",
    "5-methylundecane",
    "4-ethyldecane",
    "5,6-dimethyldecane",
    "4-ethyl-5-methylnonane",
    "4,5-diethyloctane",
    "NaCl"
]

CWchoices=[
    "1-chlorohexane",
    "2-chlorohexane",
    "3-chlorohexane",
    "Na"
]


from RadarGraph import *

import numba

###################################################Loading Data########################################################
calc_return = default_obj = lambda x: x.Reward.sum()/x.Done.sum()
worst_obj = lambda x: -x.Reward.sum()/x.Done.sum()
max_obj = lambda x:x.Reward.max()

def load_rollouts(env: str, obj=default_obj, last: bool = True, verbose: bool = False, TOL: float = 1e-4):
    """
    Retrieve RL rollouts of each algorithm from the file system

    Args:
    - env (str): The environment you want rollouts from
    - obj (function): Method to measure how good the rollout is. When set to none all rollouts are concatenated into one dataframe.
    - last (bool): Use the rollout of the last timestep if true and the best performing timestep if false
    - verbose (bool): Set to true if you want filepaths and objective evaluations printed
    - TOL (float): Tolerance for evaluating two runs as the same objective-wise (in which case the run with more episodes is preferred)

    Returns:
    - rollouts (list): List of rollouts where each element is a Pandas dataframe
    """
    #nt is for windows
    delim = "\\" if os.name=='nt' else "/"
    
    folder = f"MODELS{delim}{env}" if len(env.split(delim))==1 else env
    algoidx = [i for i,string in enumerate(folder.split(delim)) if "-v" in string][0]
    data = dict()
    all_obj=dict()
    rollName=["best_rollout","rollout"][last]
    for a,b,c in os.walk(folder):
        if rollName in c:
            algo = a.split(delim)[algoidx+1]
            df1 = pd.read_pickle(a+delim+rollName)
            if obj is not None:
                all_obj[algo] = all_obj.get(algo,[])+[obj(df1)]
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
    return data,all_obj



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
        
        
def merge_varying_graphs(folder: str = "./RL_rollout/MODELS/FictReact-v2/TD3",
                         steps: int = 100,
                         verbose: bool = False,
                         separate_runs: bool = True):
    """
    Collects results from multiple runs and merges them into one run.

    Args:
    - folder (str): Directory in which to look for monitor.csv files.
    - steps (int): Size of the step window. For fixed length environments set this to a multiple of the episode length.
    - verbose (bool): Whether or not to print additional details.
    - separate_runs (bool): Whether to calculate results for each run separately.

    Returns:
    - returns (ndarray): Shape [runs,timesteps/steps] array of average return vs step for each run.
    - episode_counts (ndarray): [runs,timesteps/steps] array of episode counts for each step window and run.
    """
    tot=0
    best=-1e10
    bestdir=""
    RESULT=np.zeros([1,2])
    all_results=[]
    for a,b,c in os.walk(folder):
        if len(c)>0:
            
            if "rollout" in c and verbose:
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

def mean_stdv_step_n(data, interp_steps=1, steps=20):
    """
    Computes the mean, standard deviation, and step counts for an array of K runs
    
    Args:
    - data (np.array):     Shape [K,totalsteps] array of runtime information
    - interp_steps (int):  How many steps to average over in axis 1 of data
    - steps (int):         The number of timesteps per single value in axis 1 of data
    
    Returns:
    - mean (np.array): Shape [totalsteps/interp_steps] array of mean values
    - stdv (np.array): Shape [totalsteps/interp_steps] array of standard deviations
    - steps (np.array): Shape [totalsteps/interp_steps] array containing what step each point is at
    
    """
    
    alt_data=data.reshape([data.shape[0],data.shape[1]//interp_steps,interp_steps])
    alt_data=alt_data.transpose(0,2,1)
    alt_data=alt_data.reshape([data.shape[0]*interp_steps,data.shape[1]//interp_steps])
    mean=alt_data.mean(axis=0)
    stdv=(alt_data.var(axis=0))**0.5
    steps=np.arange(mean.shape[0])*steps*interp_steps
    N=alt_data.shape[0]
    return mean,stdv,steps,N



def target_subset(frame, N, i):
    """
    Filters the rollout for episodes that have a specific target.

    Args:
    - frame (DataFrame): Pandas DataFrame containing gym information.
    - N (int): Number of targets in your environment.
    - i (int): The index of your target as it appears in the observation space.

    Returns:
    - cframe (DataFrame): Subset of your Pandas DataFrame with only episodes of target i.
    """
    
    obs = np.stack(frame.InState)
    #for environments with multiple vessels
    if len(obs.shape)>2:
        obs=obs[:,0,:]
        
    cframe=frame[obs[:,-N+i]>0.9]
    
    cframe=pd.concat([cframe],ignore_index=True)
    
    return cframe



def actions_by_time(frame):
    """Gives the mean action at each timestep of your rollout dataframe"""
    min_t,max_t = frame.Step.min(),frame.Step.max()
    mean_act=[]
    for t in range(min_t,max_t+1):
        mean_act+=[frame.Action[frame.Step==t].mean()]
    return np.array(mean_act)








def get_conditional_rewards(frame, targets=CWtargs):
    """
    Returns the returns conditioned on different targets.

    Args:
    - frame (DataFrame): Pandas DataFrame containing gym information.
    - targets (list, optional): List of N targets (reaction products). Defaults to CWtargs.

    Returns:
    - targets (list): List of N targets (reaction products).
    - rew (list of float): List of size N containing the average return given each target.
    """
    # turn observation column into a numpy array
    obs = np.stack(frame.InState)
    
    if len(obs.shape)>2:
        obs=obs[:,0,:]
    N=len(targets)
    rew=[]
    for i in range(N):
        #gather all data where the target is targets[N]
        cframe=frame[obs[:,-N+i]>0.9]
        #Obtain the mean reward of these episodes
        rew+=[calc_return(cframe)]
    return [targets,np.array(rew)]

def get_conditional_actions(frame, targets=CWtargs):
    """
    Returns the actions conditioned on different targets for continuous action spaces.

    Args:
    - frame (DataFrame): Pandas DataFrame containing gym information.
    - targets (list, optional): List of N targets (reaction products). Defaults to CWtargs.

    Returns:
    - targets (list): List of N targets (reaction products).
    - act (list of array): List of size N containing the mean action given each target.
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




def get_discrete_actions(frame, N=None, N2=None):
    """
    Returns the distribution of actions taken (index 0) and the average value of actions at index 1.

    Args:
    - frame (DataFrame): Pandas DataFrame containing gym information.
    - N (int, optional): Number of elements to keep in the distribution. Defaults to None.
    - N2 (int, optional): Number of elements to keep in the running average. Defaults to None.

    Returns:
    - act0 (list of float): Distribution of actions taken (index 0).
    - act1 (list of float): Average value of actions at index 1.
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

def stat_show(radar_info: dict, spoke_labels: list, labels, figsize=(22, 7), 
              gridlines=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0], relative=True, rmax=1):
    """
    Radar Plots a set of stats.

    Args:
    - radar_info (dict): A dictionary containing the data for all the radar graphs, it should be formatted like so:
                    {graph name: graph data} where graph data is a 2D array of shape [len(labels), len(spoke_labels)]
    - spoke_labels (list): A list of text labels for each spoke (vertex) of the radar graph.
    - labels (list): A list of text labels telling you what each shaded area means 
                     (e.g. the first shaded area is Return).
    - figsize (tuple): How large to make the figure. Default is (22, 7).
    - relative (bool): Whether or not to scale the last graph 
                       (Are the initial graphs 'relative' to the final graph?). Default is True.
    - gridlines (list): list of where to put each gridline. Default is [0.0, 0.4, 0.8, 1.2, 1.6, 2.0].
    
    - rmax (float): Default uppder bound to use for the range
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
            ax.set_rmax(max([a[0].max() for (c,a) in info[1:-1]]+[rmax]))
        axs.flat[-1].set_rmin(0)
    else:
        #scale them all the same
        for ax in axs.flat:
            ax.set_rmin(min([a[0].min() for (c,a) in info[1:]]+[0]))
            ax.set_rmax(max([a[0].max() for (c,a) in info[1:]]))
    
    return fig,axs