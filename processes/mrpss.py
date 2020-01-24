from type_package import *
from mp import MP
from mrp import MRP

class MRPss(MP):
    
    def __init__(self, 
                 data: Mapping[S, Mapping[S, Tuple[float, float]]], 
                 gamma: float) -> None:
        
        # dictionaries to store data
        stp = {}
        rew = {}
        for state in data.keys():
            stp[state] = {}
            rew[state] = {}
            for state2 in data[state].keys():
                stp[state][state2] = data[state][state2][0]
                rew[state][state2] = data[state][state2][1]
        super().__init__(stp)
        self.rewards_ = rew
        self.gamma_ = gamma
        # a list to preserve order for transition matrix
        self.nt_states_ = self.get_nt_states()
        # same order as nt states
        self.rmat_ = self.rewards_mat()
 
    def change_reward(self, state: S, state2: S, r: float) -> None:
        self.rewards_[state][state2] = r
        
    def remove_state(self, state: S, inbound: List[Tuple[S, Mapping[S, float]]]) -> None:
        super().remove_state(state, inbound)
        self.rewards_.pop(state, None)
        
    def add_state(self, state: S, out: Mapping[S, float], 
                  inbound: List[Tuple[S, Mapping[S, float]]],
                  r: Mapping[S, float]) -> None:
        super().add_state(state, out, inbound)
        self.rewards_[state] = r
        
    def rewards_mat(self) -> np.ndarray:
        dim = len(self.states_)
        mat = np.zeros((dim, dim))
        for i, state in enumerate(self.states_):
            for j, state2 in enumerate(self.states_):
                if state2 in self.rewards_[state]:
                    mat[i][j] = self.rewards_[state][state2]
                else:
                    pass
        return mat
    
    def get_terminal_states(self) -> Set[S]:
        term_states = super().sink_states()
        for s in term_states:
            term = False
            for state2 in self.rewards_[s]:
                if self.rewards_[s][state2] != 0:
                    term = True
            if term:
                term_states.remove(s)
                
        return term_states
    
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
    
    def to_MRPr(self) -> MRP:
        
        input_dict = {}
        for state in self.states_:
            r = 0
            for state2 in self.rewards_[state].keys():
                r += self.stp_[state][state2] * self.rewards_[state][state2]
            input_dict[state] = (self.stp_[state], r)
        return MRP(input_dict, self.gamma_)

if __name__ == '__main__':
    data = {
        1: {1: (0.6, 7), 2: (0.3, 7), 3: (0.1, 7)},
        2: {1: (0.1, 10), 2: (0.2, 10), 3: (0.7, 10)},
        3: {3: (1.0, 0)}
    }
    mrp_obj = MRPss(data, 1.0)
    print(mrp_obj.tr_matrix())
    print()
    print(mrp_obj.rmat_)
    print()
    terminal = mrp_obj.get_terminal_states()
    print(terminal)
    print()

    mrp = mrp_obj.to_MRPr()
    print(mrp.tr_matrix())
    print()
    print(mrp.rvec_)
    print()
    terminal = mrp.get_terminal_states()
    print(terminal)
    print()