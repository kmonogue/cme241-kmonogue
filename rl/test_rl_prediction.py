import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/kevinmonogue/cme241-kmonogue/')

import time
from rl_base import *
from mdp_RL_interface import *
from dp_algo.policy_eval import *

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

    mdp_obj = MDPrl(mdp_data, 1)

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
    print('Dynamic Programming')
    start = time.time()
    print(policy_eval(mdp_obj, pol_obj, 0.001))
    end = time.time()
    print(end - start)
    print()

    possible_actions = {}
    for i in range(16):
        possible_actions[i] = ['n', 's', 'w', 'e']

    rl = RL(possible_actions, 1, 0.1, mdp_obj.reward, mdp_obj.next_state, mdp_obj.get_terminal_states())
    start = time.time()
    rl.evaluate_TD_v(4, policy_data, 0, 100000, 0.01)
    end = time.time()
    print('TD(0)')
    print(rl.vf)
    print(end - start)
    print()

    rl = RL(possible_actions, 1, 0.1, mdp_obj.reward, mdp_obj.next_state, mdp_obj.get_terminal_states())
    start = time.time()
    for i in range(5000):
        path = rl.generate_MC_path(np.random.randint(0, 16), policy_data, 1000)
        rl.evaluate_MC_episode_v(path, rl.alpha)
    end = time.time()
    print('MC')
    print(rl.vf)
    print(end - start)
    print()

    rl = RL(possible_actions, 1, 0.1, mdp_obj.reward, mdp_obj.next_state, mdp_obj.get_terminal_states())
    start = time.time()
    for i in range(5000):
        path = rl.generate_MC_path(np.random.randint(0, 16), policy_data, 1000)
        rl.evaluate_TD_lambda_episode_v(path, rl.alpha, 0.9)
    end = time.time()
    print('TD Lambda Forward')
    print(rl.vf)
    print(end - start)
    print()

    rl = RL(possible_actions, 1.0, 0.1, mdp_obj.reward, mdp_obj.next_state, mdp_obj.get_terminal_states())
    start = time.time()
    rl.evaluate_TD_lambda_backward_v(4, policy_data, 0, 100000, 0.01, 0.9)
    end = time.time()
    print('TD Lambda Backward')
    print(rl.vf)
    print(end - start)
    print()