import importlib
import json
from typing import NamedTuple, Tuple, Callable, Optional, List
import gymnasium as gym

def make_obj(entry_point, kwargs):
    mod_name, attr_name = entry_point.split(":")
    #Build the MultiDiscrete gym
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)(**kwargs)

def serial(x):
    """Turn a numpy array into tuples"""
    try:return float(x)
    except:return tuple(a for a in x)

def save_config(kwargs, fn):
    with open(fn,"w") as f:
        f.write(json.dumps(kwargs, indent=2, default=serial))


class PolicyParam(NamedTuple):
    name: str
    entry_point: str
    args: dict
    bench_idx: int

class BenchParam(NamedTuple):
    name: str
    gym_id: str

class ManagerAction(NamedTuple):
    name: str
    args: dict
    cost: float
    bench: Optional[int]

def add_policy_to_list(policies: list, policy: PolicyParam):
    instance = make_obj(policy.entry_point,policy.args)
    policies[policy.bench_idx] += [(policy.name, instance)]

def parse_json(fn):
    with open(fn,"r") as f:
        kwargs = json.load(f)

    bench_info = [BenchParam(*args) for args in kwargs["benches"]]
    benches=[]
    bench_names = []
    for benchparam in bench_info:
        benches.append(gym.make(benchparam.gym_id))
        bench_names.append(benchparam.name)

    policies = [[] for _ in bench_info]

    for args in kwargs["policies"]:
        add_policy_to_list(policies, PolicyParam(*args))

    targets = kwargs["targets"]

    actions = [ManagerAction(*args) for args in kwargs["actions"]]


    return benches, bench_names, policies, targets, actions