from processes.type_package import *
from processes.mp import MP

class MRP(MP):
    
    def __init__(self, 
                 data: Mapping[S, Tuple[Mapping[S, float], float]], 
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

if __name__ == '__main__':
    data = {
        1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
        2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
        3: ({3: 1.0}, 0.0)
    }
    mrp_obj = MRP(data, 1.0)
    print(mrp_obj.tr_matrix())
    print()
    print(mrp_obj.rvec_)
    print()
    terminal = mrp_obj.get_terminal_states()
    print(terminal)