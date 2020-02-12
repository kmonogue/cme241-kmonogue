import sys
sys.path.append('/Users/kevinmonogue/cme241-kmonogue/')

from processes.mdpss import MDP
from processes.policy import Policy
from processes.vf import VF
from dp_algo.policy_evalss import *
import numpy as np
import copy

def value_iter(mdp: MDP, vf: VF, tol: float) -> VF:

    v_old = np.ones(len(mdp.states_))
    v_new = np.zeros(len(v_old))
    count = 0
    while (np.linalg.norm(v_old - v_new) > tol):
        count += 1
        v_old = v_new
        for state in mdp.states_:
            max_val = float('-inf')
            for action in mdp.s_a_s_[state]:
                val = 0
                for state2 in mdp.s_a_s_[state][action]:
                    prob_move = mdp.s_a_s_[state][action][state2][0]
                    future_val = vf.value_dict_[state2]
                    current_reward = mdp.s_a_s_[state][action][state2][1]
                    '''if count == 2 and state == 2:
                        print(state2)
                        print(future_val)
                        print(prob_move)
                        print(current_reward) '''
                    val += prob_move * (mdp.gamma_ * future_val + current_reward)
                if val > max_val:
                    max_val = val
                '''if count == 2 and state == 2:
                    print(action)
                    print(val)
                    print()'''

            vf.value_dict_[state] = max_val
        v_new = vf.get_vector(list(mdp.states_))
        #print(vf.value_dict_[1])
        #print(vf.value_dict_[2])

    return vf

if __name__ == '__main__':
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
                'w': ({0: 1.0}, -1.0),
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
                'n': ({0: 1.0}, -1.0),
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
                's': ({15: 1.0}, -1.0),
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
                'e': ({15: 1.0}, -1.0),
                'w': ({13: 1.0}, -1.0),
            },
            15: {
                'n': ({15: 1.0}, 0.0),
                's': ({15: 1.0}, 0.0),
                'e': ({15: 1.0}, 0.0),
                'w': ({15: 1.0}, 0.0),
            },
        }

    mdp = MDP(mdp_data, 1.0)

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

    policy = Policy(policy_data)
    vf = policy_eval(mdp, policy, 0.001)

    print(value_iter(mdp, vf))
