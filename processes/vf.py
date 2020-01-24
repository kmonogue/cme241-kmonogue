from type_package import *

class VF(Generic[S]):
    
    def __init__(self, data: Mapping[S, float] = None, 
                 vec: np.array = None, states: List[S] = None) -> None:
        if vec is None or states is None:
            self.value_dict_ = data
        else:
            self.value_dict_ = self.produce_dict(vec, states)
        
    def get_value(self, state: S) -> float:
        return self.value_dict_data
        
    def get_vector(self, states: List[S]) -> np.array:
        vec = np.zeros(len(states))
        for i, state in enumerate(states):
            vec[i] = self.value_dict_[state]
        return vec
    
    def produce_dict(self, vec: np.array, states: List[S]) -> Mapping[S, float]:
        result = {}
        for i, state in enumerate(states):
            result[state] = vec[i]
        return result
        
    def __repr__(self):
        return self.value_dict_.__repr__()

    def __str__(self):
        return self.value_dict_.__str__()