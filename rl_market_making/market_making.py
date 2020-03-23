import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/kevinmonogue/cme241-kmonogue/')

import time
from rl.rl_base import *
from rl.mdp_RL_interface import *
from dp_algo.policy_iter import *
from processes.mdp import *
from processes.type_package import *
import numpy as np
import timeit
import matplotlib.pyplot as plt

# first we initialize our set of states and add to our mdp data
# the type of our start is Tuple[time, Stock, Wealth, Inventory]
def generate_samples(time_steps, price_low, price_high, price_steps, num_shares, max_spread):

    #initialize other scalars
    price_step_val = (price_high - price_low) / price_steps
    max_inv = num_shares * time_steps
    max_wealth = max_inv * (price_high + max_spread)
    min_wealth = max_inv * (price_low - max_spread)
    num_states = time_steps * price_steps * ((max_wealth + min_wealth) / price_step_val) * (max_inv * 2 + 1)
    num_actions = ((max_spread / price_step_val) ** 2) * (num_shares ** 2)

    start = timeit.default_timer()
    shares_range = range(num_shares + 1)
    state_action = {}
    t_states = set()
    counter = 1
    for time in range(time_steps + 1):
        for price in np.arange(price_low, price_high + price_step_val, price_step_val):
            price = round(price, 2)
            bid_range = np.arange(price - max_spread, price, price_step_val)
            ask_range = np.arange(price, price + max_spread + price_step_val, price_step_val)

            for wealth in np.arange(-min_wealth, max_wealth + price_step_val, price_step_val):
                wealth = round(wealth, 2)
                for inv in range(-max_inv, max_inv + 1):
                    state = (time, price, wealth, inv)
                    if time == time_steps:
                        t_states.add(state)
                    actions = []
                    for bid in bid_range:
                        bid = round(bid, 2)
                        for ask in ask_range:
                            ask = round(bid, 2)
                            for b in shares_range:
                                for a in shares_range:
                                    action = (bid, b, ask, a)
                                    actions.append(action)
                    state_action[state] = actions
                    
                    '''
                    if counter % 1000 == 0:
                        print(f'State:{counter}')
                        print((timeit.default_timer() - start) / counter * (num_states - counter) / 60)
                        print()
                    counter += 1 '''

    return state_action, t_states


# next we define a function to return a new state given a current state and action pair
def mm_next_state(state: S, action: A) -> S:
    
    # HARDCODED FOR OUR TABULAR STATE DESIGN
    time_steps = 3
    price_low = 0.1
    price_high = 0.2
    price_steps = 10
    max_shares = 2
    max_spread = 0.03
    
    price_step_val = (price_high - price_low) / price_steps
    max_spread_int = int(max_spread / price_step_val)
    max_inv = max_shares * time_steps
    max_wealth = max_inv * (price_high + max_spread)
    min_wealth = max_inv * (price_low - max_spread)

    # determine the number of shares bid hit
    spread = abs(int(state[1] - action[0] / price_step_val))
    num_bought = 0
    for _ in range(max_spread_int - spread + 1):
        prob = np.random.uniform()
        if prob > 0.5:
            num_bought += 1
    num_bought = min(num_bought, action[1])
    
     # determine the number of shares ask lifted
    spread = abs(int(state[1] - action[2] / price_step_val))
    num_sold = 0
    for _ in range(max_spread_int - spread + 1):
        prob = np.random.uniform()
        if prob > 0.5:
            num_sold += 1
    num_sold = min(num_sold, action[3])

    # update wealth and inventory
    wealth = state[2] + num_sold * action[2] - num_bought * action[0]

    # THIS IS HARDCODED FOR OUR TABULAR STATES
    wealth = min(max_wealth, wealth)
    wealth = max(-min_wealth, wealth)
    wealth = round(wealth, 2)

    inventory = state[3] + num_bought - num_sold
    inventory = min(max_inv, inventory)
    inventory = max(-max_inv, inventory)

    # evolve the stock price variable
    ret = np.random.normal(0, 0.015)
    new_price = round(state[1] + ret, 2)
    new_price = max(price_low, new_price)
    new_price = min(price_high, new_price)

    return (state[0] + 1, new_price, wealth, inventory)

# next we define a function to return a new state given a current state and action pair
def mm_reward(state: S, action: A) -> float:
    
    # hard coded for this example
    terminal = 3
    
    # our terminal state is actually a dummy state, so we reward one state prior
    if state[0] == terminal - 1:
        try:
            price = 0
            if state[3] > 0:
                price = state[1] * 0.95
            else:
                price = state[1] * 1.05
            x = state[2] + state[3] * price
            return (x ** 0.1 - 1) / 0.1
        except:
            print(state[2])
            print(state[3])
            print(state[1])
            return 0
    else:
        return 0

time_steps = 3
price_low = 0.1
price_high = 0.2
price_steps = 10
max_shares = 2
max_spread = 0.03

state_action, t_states = generate_samples(time_steps, price_low, price_high, price_steps, max_shares, max_spread)

'''
for state in state_action.keys():
    if state[0] == 0:
        if state[2] > 0.98:
            if state[2] < 1.02:
                print(state)
                '''


rl = RL(state_action, 1, 0.1, mm_reward, mm_next_state, t_states)
start = time.time()
rl.sarsa(initial = (0, 0.15, 0, 0), length_param = 0, iters = 10000, alpha = 0.1, eps = 0.1)
end = time.time()
print('Sarsa')
pol = rl.derived_pol()
pb = []
b = []
ps = []
s = []
w = []
stock = []
for state in state_action.keys():
    if state[0] == 2:
        action = list(pol[state].keys())[0]
        b.append(action[1])
        pb.append(state[1] - action[0])
        s.append(action[3])
        ps.append(action[2] - state[1])
        w.append(state[2])
        stock.append(state[1])

plt.subplot(3, 2, 1)
plt.scatter(pb, b)
plt.title('Shares bought by bid spread')

plt.subplot(3, 2, 2)
plt.scatter(ps, s)
plt.title('Shares sold by ask spread')

plt.subplot(3, 2, 3)
plt.scatter(w, b)
plt.title('Shares bought by wealth')

plt.subplot(3, 2, 4)
plt.scatter(w, s)
plt.title('Shares sold by wealth')

plt.subplot(3, 2, 5)
plt.scatter(stock, b)
plt.title('Shares bought by stock price')

plt.subplot(3, 2, 6)
plt.scatter(stock, s)
plt.title('Shares sold by stock price')

plt.tight_layout()

plt.savefig('Sarsa.png')
print(end - start)
print()
plt.clf()

rl = RL(state_action, 1, 0.1, mm_reward, mm_next_state, t_states)
start = time.time()
rl.q_learn(initial = (0, 0.15, 0, 0), length_param = 0, iters = 10000, alpha = 0.1, eps = 0.1)
end = time.time()
print('Q-Learn')
pol = rl.derived_pol()
pb = []
b = []
ps = []
s = []
w = []
stock = []
for state in state_action.keys():
    if state[0] == 2:
        action = list(pol[state].keys())[0]
        b.append(action[1])
        pb.append(state[1] - action[0])
        s.append(action[3])
        ps.append(action[2] - state[1])
        w.append(state[2])
        stock.append(state[1])

plt.subplot(3, 2, 1)
plt.scatter(pb, b)
plt.title('Shares bought by bid spread')

plt.subplot(3, 2, 2)
plt.scatter(ps, s)
plt.title('Shares sold by ask spread')

plt.subplot(3, 2, 3)
plt.scatter(w, b)
plt.title('Shares bought by wealth')

plt.subplot(3, 2, 4)
plt.scatter(w, s)
plt.title('Shares sold by wealth')

plt.subplot(3, 2, 5)
plt.scatter(stock, b)
plt.title('Shares bought by stock price')

plt.subplot(3, 2, 6)
plt.scatter(stock, s)
plt.title('Shares sold by stock price')

plt.tight_layout()

plt.savefig('Q-learn.png')
print(end - start)


