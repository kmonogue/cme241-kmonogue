from type_package import *

class MP(Generic[S]):
    
    # stp = state transition probability
    def __init__(self, stp: Mapping[S, Mapping[S, float]] = None) -> None:
        self.stp_ = stp
        # a list to preserve order for transition matrix
        self.states_ = stp.keys()
    
    def get_transitions(self, state: S) -> Mapping[S, float]:
        return self.stp_[state]
    
    def get_inbound(self, state: S) -> Set[S]:
        
        states = []
        for in_state in self.stp_.keys():
            if state in self.stp_[in_state].keys():
                states.append(in_state)
                
        return states
    
    def update_transitions(self, state: S, transitions: Mapping[S, float]) -> None:
        
        total = 0
        for v in transitions.values():
            assert v >= 0, "Probabilities must be non-negative"
            total += v
        assert total == 1, "Probabilities must sum to 1"
        self.stp_[state] = transitions
        
    def add_state(self, state: S, out: Mapping[S, float], 
                  inbound: List[Tuple[S, Mapping[S, float]]]) -> None:
        
        self.update_transitions(state, out)
        for s, tr in inbound:
            self.update_transitions(s, tr)
        self.states_.add(state)
        
    def remove_state(self, state: S, inbound: List[Tuple[S, Mapping[S, float]]]) -> None:
        
        inbound_states = set(self.get_inbound(state))
        arg_states = set()
        for s, tr in inbound:
            arg_states.add(s)
        assert arg_states == inbound_states, "Incorrect replacement set of inbound states"
        
        for s, tr in inbound:
            self.update_transitions(s, tr)
            
        self.states_.remove(state)
        
    def tr_matrix(self) -> np.ndarray:
        
        dim = len(self.states_)
        mat = np.zeros((dim, dim))
        for i, state1 in enumerate(self.states_):
            for j, state2 in enumerate(self.states_):
                try:
                    mat[i][j] = self.stp_[state1][state2]
                except:
                    mat[i][j] = 0
        return mat
    
    def sink_states(self) -> Set[S]:
        states = set()
        for k in self.stp_.keys():
            if len(self.stp_[k]) == 1:
                states.add(k)
        return states

if __name__ == "__main__":
    transitions = {
        1: {1: 0.1, 2: 0.6, 3: 0.1, 4: 0.2},
        2: {1: 0.25, 2: 0.22, 3: 0.24, 4: 0.29},
        3: {1: 0.7, 2: 0.3},
        4: {1: 0.3, 2: 0.5, 3: 0.2},
        5: {5: 1.0}
    }
    mp_obj = MP(transitions)
    print(mp_obj.stp_)
    print(mp_obj.states_)
    print(mp_obj.sink_states())