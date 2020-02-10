from processes.type_package import *
from processes.mrp import MRP
from processes.mrpv import MRPv
from processes.policy import Policy

class MDP(Generic[S, A]):
    
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
        # a list to preserve order for transition matrix
        self.nt_states_ = self.get_nt_states()
 
    def change_reward(self, state: S, action: A, r: float) -> None:
        self.s_a_r_[state][action] = r
        
    def remove_state(self, state: S, 
                     inbound: List[Tuple[S, Mapping[A, Tuple[Mapping[S, float], float]]]]) -> None:
        self.states_.remove(state)
        for state, actions in inbound:
            for action in actions.keys():
                self.s_a_s_[state][action] = actions[action][0]
                self.s_a_r_[state][action] = actions[action][1]
        self.s_a_s_.pop(state, None)
        self.s_a_r_.pop(state, None)
        
    def add_state(self, state: S, out: Mapping[A, Tuple[Mapping[S, float], float]], 
                  inbound: List[Tuple[S, Mapping[A, Tuple[Mapping[S, float], float]]]]) -> None:
        self.states_.add(state)
        for state, actions in out:
            for action in actions.keys():
                self.s_a_s_[state][action] = actions[action][0]
                self.s_a_r_[state][action] = actions[action][1]
                
        for state, actions in inbound:
            for action in actions.keys():
                self.s_a_s_[state][action] = actions[action][0]
                self.s_a_r_[state][action] = actions[action][1]
    
    def get_sink_states(self) -> Set[S]:
        sink = set()
        for state in self.s_a_s_.keys():
            for action in self.s_a_s_[state].keys():
                if len(self.s_a_s_[state][action].keys()) == 1 and state in self.s_a_s_[state][action].keys():
                    sink.add(state)
        return sink
    
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
    
    def get_nt_states(self) -> List[S]:
        t_states = self.get_terminal_states()
        return [s for s in self.states_ if s not in t_states]
    
    def to_mrp(self, pi: Policy) -> MRP:
        
        # the goal here is to produce the input to the MRP constructor
        mrp_input =  {}
        for state in self.s_a_s_.keys():
            output_states = set()
            output_reward = 0
            for action in pi.get_actions(state).keys():
                output_states = output_states.union(set(self.s_a_s_[state][action].keys()))
                output_reward += self.s_a_r_[state][action] * pi.get_prob(state, action)
            
            output_probs = {}
            
            for state2 in output_states:
                for action in pi.get_actions(state).keys():
                    if state2 in self.s_a_s_[state][action].keys():
                        if state2 in output_probs.keys():
                            output_probs[state2] += self.s_a_s_[state][action][state2] * pi.get_prob(state, action)
                        else:    
                            output_probs[state2] = self.s_a_s_[state][action][state2] * pi.get_prob(state, action)
                       
            
            mrp_input[state] = (output_probs, output_reward)
        
        return MRP(mrp_input, self.gamma_)
    
    def to_mrpv(self, pi: Policy) -> MRPv:
        
        # the goal here is to produce the input to the MRP constructor
        mrp_input =  {}
        for state in self.s_a_s_.keys():
            output_states = set()
            output_reward = 0
            for action in pi.get_actions(state).keys():
                output_states = output_states.union(set(self.s_a_s_[state][action].keys()))
                output_reward += self.s_a_r_[state][action] * pi.get_prob(state, action)
            
            output_probs = {}
            
            for state2 in output_states:
                for action in pi.get_actions(state).keys():
                    if state2 in self.s_a_s_[state][action].keys():
                        if state2 in output_probs.keys():
                            output_probs[state2] += self.s_a_s_[state][action][state2] * pi.get_prob(state, action)
                        else:    
                            output_probs[state2] = self.s_a_s_[state][action][state2] * pi.get_prob(state, action)
                       
            
            mrp_input[state] = (output_probs, output_reward)
        
        return MRPv(mrp_input, self.gamma_)
    
if __name__ == '__main__':
    data = {
    1: {
        'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
        'b': ({2: 0.3, 3: 0.7}, 2.8),
        'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
    2: {
        'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
        'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
    3: {
        'a': ({3: 1.0}, 0.0),
        'b': ({3: 1.0}, 0.0)
        }
    }

    mdp_obj = MDP(data, 0.95)
    print(mdp_obj.states_)
    print()
    print(mdp_obj.s_a_s_)
    print()
    print(mdp_obj.s_a_r_)
    print()
    terminal = mdp_obj.get_terminal_states()
    print(terminal)
    print()

    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }

    pol_obj = Policy(policy_data)
    mdp_data = {
        1: {
            'a': ({1: 0.2, 2: 0.6, 3: 0.2}, 7.0),
            'b': ({1: 0.6, 2: 0.3, 3: 0.1}, -2.0),
            'c': ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0)
        },
        2: {
            'a': ({1: 0.1, 2: 0.6, 3: 0.3}, 1.0),
            'c': ({1: 0.6, 2: 0.2, 3: 0.2}, -1.2)
        },
        3: {
            'b': ({3: 1.0}, 0.0)
        }
    }
    mdp1_obj = MDP(mdp_data, gamma=0.9)
    mrp_obj = mdp1_obj.to_mrp(pol_obj)
    print(mrp_obj.stp_)
    print()
    print(mrp_obj.rewards_)
    print()
    print(mrp_obj.tr_matrix())
    print()
    print(mrp_obj.rvec_)
    print()
    terminal = mrp_obj.get_terminal_states()
    print(terminal)