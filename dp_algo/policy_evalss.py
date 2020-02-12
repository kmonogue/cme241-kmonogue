import sys
sys.path.append('/Users/kevinmonogue/cme241-kmonogue/')

from processes.mdpss import MDP
from processes.policy import Policy
from processes.vf import VF
import numpy as np
import copy

def policy_eval(mdp: MDP, policy: Policy, tol: float) -> VF:

    # initialize containers
    vf = {}
    v_old = np.ones(len(mdp.states_))
    v_new = np.zeros(len(mdp.states_))

    # initialize values to zero
    for state in mdp.states_:
        vf[state] = 0

    # while not converged
    while max(np.abs(v_new - v_old)) > tol:
        v_old = copy.deepcopy(v_new)

        #for each state
        for i, state in enumerate(mdp.states_):
            next_val = 0

            # find value of each action
            for action in policy.s_a_prob_[state].keys():
                action_prob = policy.s_a_prob_[state][action]
                action_val = 0

                # value is sum across all end states
                # immediate reward + discounted future reward
                for state2 in mdp.s_a_s_[state][action].keys():
                    prob_move = mdp.s_a_s_[state][action][state2][0]
                    move_reward = mdp.s_a_s_[state][action][state2][1]
                    move_future = vf[state2]
                    action_val += prob_move * (move_reward + mdp.gamma_ * move_future)

                # sum across all actions
                next_val += action_prob * (action_val)
                
            # update values
            v_new[i] = next_val
            vf[state] = next_val
            
    return VF(vf)



if __name__ == "__main__":
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

    mdp_obj = MDP(mdp_data, 1)
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
    
    mdp1_obj = MDP(mdp_data, gamma=1)
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
    print(policy_iter(mdp_obj, pol_obj, 0.001))
    print(mrp_obj.solve_value())
