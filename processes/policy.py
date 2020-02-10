from processes.type_package import *

class Policy(Generic[S, A]):
    
    def __init__(self, data: Mapping[S, Mapping[A, float]]) -> None:
        for state in data.keys():
            count = 0
            for action in data[state].keys():
                assert data[state][action] >= 0, "probability must be positive"
                assert data[state][action] <= 1, "probability must be less than 1"
                count += data[state][action]
            assert count == 1, "probabilities must sum to one"
        self.s_a_prob_ = data
        
    def update_prob(self, state: S, action: A, prob: float) -> None:
        assert prob >= 0, "probability must be positive"
        assert prob <= 1, "probability must be less than 1"
        count = prob
        for action2 in self.s_a_prob_[state].keys():
            if action2 == action:
                pass
            else:
                count += self.s_a_prob_[state][action]
        assert count == 1, "probabilities must sum to one"
        
        self.s_a_prob_[state][action] = prob
        
    def get_prob(self, state: S, action: A) -> float:
        return self.s_a_prob_[state][action]
    
    def get_actions(self, state: S) -> Mapping[A, float]:
        return self.s_a_prob_[state]
        
    def __repr__(self):
        return self.s_a_prob_.__repr__()

    def __str__(self):
        return self.s_a_prob_.__str__()