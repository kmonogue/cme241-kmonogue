import sys
sys.path.append('/Users/kevinmonogue/cme241-kmonogue/')

from processes.mdp import MDP
from processes.policy import Policy
from processes.vf import VF
from dp_algo.policy_eval import *
import numpy as np
import copy

def policy_improve(mdp: MDP, vf: VF) -> Policy:
    
    new_pol = {}
    for state in mdp.states_:
        max_val = float('-inf')
        max_action = []
        for action in mdp.s_a_s_[state].keys():
            action_val = 0
            for state2 in mdp.s_a_s_[state][action].keys():
                action_val += mdp.s_a_s_[state][action][state2] * (mdp.s_a_r_[state][action] + mdp.gamma_ * vf.get_value(state2))
            if action_val > max_val:
                max_val = action_val
                max_action = [action]
            elif action_val == max_val:
                max_action.append(action)
        actions = {}
        for action in max_action:
            actions[action] = 1.0 / len(max_action)
        new_pol[state] = actions
    
    return Policy(new_pol)

def policy_iter(mdp: MDP, policy: Policy, tol: float) -> (VF, Policy):
    
    vf = policy_eval(mdp, policy, tol)
    v_old = vf.get_vector(list(mdp.states_))
    v_new = np.ones(len(v_old))
    while (max(abs(v_new - v_old)) > tol):
        v_old = v_new
        new_pol = policy_improve(mdp, vf)
        new_vf = policy_eval(mdp, new_pol, tol)
        v_new = new_vf.get_vector(list(mdp.states_))

    return new_vf, new_pol


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

    pol = Policy(policy_data)
    #vf = policy_eval(mdp, pol, 0.001)
    #new_pol = policy_improve(mdp, pol, vf)
    #print(new_pol)
    vf, pol = policy_iter(mdp, pol, 0.001)
            
