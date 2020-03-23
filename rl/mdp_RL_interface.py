from processes.type_package import *
from processes.mrp import MRP
from processes.mrpv import MRPv
from processes.policy import Policy
import numpy as np

class MDPrl(Generic[S, A]):
    
    def __init__(self, 
                 data: Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]], 
                 gamma: float) -> None:
        
        # dictionaries to store data
        s_a_s = {}
        s_a_r = {}
        for state in data.keys():
            s_a_s[state] = {}
            s_a_r[state] = {}
            for action in data[state].keys():
                s_a_s[state][action] = data[state][action][0]
                s_a_r[state][action] = data[state][action][1]
        self.states_ = set(data.keys())
        self.s_a_s_ = s_a_s
        self.s_a_r_ = s_a_r
        self.gamma_ = gamma
        self.t_states_ = self.get_terminal_states()
    
    def get_sink_states(self) -> Set[S]:
        sink_set = set()
        for state in self.s_a_s_.keys():
            sink = True
            for action in self.s_a_s_[state].keys():
                if len(self.s_a_s_[state][action]) == 1 and state in self.s_a_s_[state][action].keys():
                    continue
                else:
                    sink = False
            if sink:
                sink_set.add(state)
        return sink_set
    
    def get_terminal_states(self) -> Set[S]:
        sink_states = self.get_sink_states()
        term_states = set()
        zero_r = True
        for state in sink_states:
            for action in self.s_a_r_[state].keys():
                if self.s_a_r_[state][action] != 0:
                    zero_r = False
            if zero_r:
                term_states.add(state)
            
        return term_states
    
    def next_state(self, state: S, action: A) -> S:
        rand = np.random.uniform()
        cum_prob = 0
        for state2 in self.s_a_s_[state][action].keys():
            cum_prob += self.s_a_s_[state][action][state2]
            if cum_prob >= rand:
                return state2

    def reward(self, state: S, action: A) -> float:
        return self.s_a_r_[state][action]
    

     