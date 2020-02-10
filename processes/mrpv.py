from processes.type_package import *
from processes.mp import MP
from processes.vf import VF
from processes.policy import Policy
#for testing only
#from mdp import MDP

class MRPv(MP):
    
    def __init__(self, data: Mapping[S, Tuple[Mapping[S, float], float]], 
                 gamma: float) -> None:
        
        # dictionaries to store data
        stp = {}
        rew = {}
        for state in data.keys():
            stp[state] = data[state][0]
            rew[state] = data[state][1]
        super().__init__(stp)
        self.rewards_ = rew
        self.gamma_ = gamma
        # a list to preserve order for transition matrix
        self.nt_states_ = self.get_nt_states()
        # same order as nt states
        self.rvec_ = self.rewards_vec()
        
        #solve for value vector
        self.values_ = self.solve_value()
        self.value_dict_ = VF(vec = self.values_, states = self.nt_states_)
 
    def change_reward(self, state: S, r: float) -> None:
        self.rewards_[state] = r
        
    def remove_state(self, state: S, inbound: List[Tuple[S, Mapping[S, float]]]) -> None:
        super().remove_state(state, inbound)
        self.rewards_.pop(state, None)
        
    def add_state(self, state: S, out: Mapping[S, float], 
                  inbound: List[Tuple[S, Mapping[S, float]]], r: float) -> None:
        super().add_state(state, out, inbound)
        self.rewards_[state] = r
        
    def rewards_vec(self) -> np.array:
        r = [0] * len(self.nt_states_)
        for i, state in enumerate(self.nt_states_):
            r[i] = self.rewards_[state]
        return np.asarray(r)
    
    def get_terminal_states(self) -> Set[S]:
        term_states = super().sink_states()
        return {s for s in term_states if self.rewards_[s] == 0}
    
    def get_nt_states(self) -> List[S]:
        t_states = self.get_terminal_states()
        return [s for s in self.states_ if s not in t_states]
    
    def tr_matrix(self) -> np.ndarray:
        
        dim = len(self.nt_states_)
        mat = np.zeros((dim, dim))
        for i, state1 in enumerate(self.nt_states_):
            for j, state2 in enumerate(self.nt_states_):
                try:
                    mat[i][j] = self.stp_[state1][state2]
                except:
                    mat[i][j] = 0
        return mat
    
    def solve_value(self) -> np.array:
        dim = len(self.nt_states_)
        p = self.tr_matrix()
        vec = np.matmul(np.linalg.inv((np.identity(dim) - self.gamma_ * p)), self.rvec_)
        return vec

if __name__ == '__main__':
    policy_data = {
        1: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        2: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        3: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        4: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        5: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        6: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        7: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        8: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        9: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        10: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        11: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        12: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        13: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        14: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        15: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
        0: {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25},
    }

    pol_obj = Policy(policy_data)
    mdp_data = {
        0: {
            'n': ({0 : 1.0}, 0.0),
            's': ({0 : 1.0}, 0.0),
            'e': ({0 : 1.0}, 0.0),
            'w': ({0 : 1.0}, 0.0),
        },
        1: {
            'n': ({1: 1.0}, -1.0),
            's': ({5: 1.0}, -1.0),
            'e': ({2: 1.0}, -1.0),
            'w': ({0: 1.0}, 0.0),
        },
        2: {
            'n': ({2: 1.0}, -1.0),
            's': ({6: 1.0}, -1.0),
            'e': ({3: 1.0}, -1.0),
            'w': ({1: 1.0}, -1.0),
        },
        3: {
            'n': ({3: 1.0}, -1.0),
            's': ({7: 1.0}, -1.0),
            'e': ({3: 1.0}, -1.0),
            'w': ({2: 1.0}, -1.0),
        },
        4: {
            'n': ({0: 1.0}, 0.0),
            's': ({8: 1.0}, -1.0),
            'e': ({5: 1.0}, -1.0),
            'w': ({4: 1.0}, -1.0),
        },
        5: {
            'n': ({1: 1.0}, -1.0),
            's': ({9: 1.0}, -1.0),
            'e': ({6: 1.0}, -1.0),
            'w': ({4: 1.0}, -1.0),
        },
        6: {
            'n': ({2: 1.0}, -1.0),
            's': ({10: 1.0}, -1.0),
            'e': ({7: 1.0}, -1.0),
            'w': ({5: 1.0}, -1.0),
        },
        7: {
            'n': ({3: 1.0}, -1.0),
            's': ({11: 1.0}, -1.0),
            'e': ({7: 1.0}, -1.0),
            'w': ({6: 1.0}, -1.0),
        },
        8: {
            'n': ({4: 1.0}, -1.0),
            's': ({12: 1.0}, -1.0),
            'e': ({9: 1.0}, -1.0),
            'w': ({8: 1.0}, -1.0),
        },
        9: {
            'n': ({5: 1.0}, -1.0),
            's': ({13: 1.0}, -1.0),
            'e': ({10: 1.0}, -1.0),
            'w': ({8: 1.0}, -1.0),
        },
        10: {
            'n': ({6: 1.0}, -1.0),
            's': ({14: 1.0}, -1.0),
            'e': ({11: 1.0}, -1.0),
            'w': ({9: 1.0}, -1.0),
        },
        11: {
            'n': ({7: 1.0}, -1.0),
            's': ({15: 1.0}, 0.0),
            'e': ({11: 1.0}, -1.0),
            'w': ({10: 1.0}, -1.0),
        },
        12: {
            'n': ({8: 1.0}, -1.0),
            's': ({12: 1.0}, -1.0),
            'e': ({13: 1.0}, -1.0),
            'w': ({12: 1.0}, -1.0),
        },
        13: {
            'n': ({9: 1.0}, -1.0),
            's': ({13: 1.0}, -1.0),
            'e': ({14: 1.0}, -1.0),
            'w': ({12: 1.0}, -1.0),
        },
        14: {
            'n': ({10: 1.0}, -1.0),
            's': ({14: 1.0}, -1.0),
            'e': ({15: 1.0}, 0.0),
            'w': ({13: 1.0}, -1.0),
        },
        15: {
            'n': ({15: 1.0}, 0.0),
            's': ({15: 1.0}, 0.0),
            'e': ({15: 1.0}, 0.0),
            'w': ({15: 1.0}, 0.0),
        },
    }
    mdp1_obj = MDP(mdp_data, gamma=0.9)
    mrp_obj = mdp1_obj.to_mrpv(pol_obj)
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
    print()
    print(mrp_obj.value_dict_)