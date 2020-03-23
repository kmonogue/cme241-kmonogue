from processes.type_package import *
from rl.mdp_RL_interface import *
import random

class RL(Generic[S, A]):
    
    # type of get_* is a function that returns 
    def __init__(self, 
                 data: Mapping[S, List[A]], 
                 gamma: float,
                 alpha: float,
                 get_reward,
                 get_transition,
                 t_states: Set[S]) -> None:
        
        # dictionaries to store data
        self.s_a = data
        vf = {}
        q_vf = {}
        for state in data.keys():
            vf[state] = 0
            q_vf[state] = {}
            for action in data[state]:
                q_vf[state][action] = 0
        self.vf = vf
        self.q_vf = q_vf
        self.derived_policy = {}
        self.gamma = gamma
        self.alpha = alpha
        self.reward = get_reward
        self.next_state = get_transition
        self.t_states = t_states
            
    def reward(self, state: S, action: A) -> float:
        return self.reward(S, A)

    def next_state(self, state: S, action: A) -> S:
        return self.next_state(S, A)

    def generate_MC_path(self, initial: S, policy: Mapping[S, Mapping[A, float]], max_steps: int) -> Tuple[List[S], List[A], List[float]]:
        states = [initial]
        actions = []
        rewards = []
        count = 0
        terminate = False
        while (not terminate) and count < max_steps:
            cur_state = states[-1]
            action = None
            rand = np.random.uniform()
            cum_prob = 0
            for action_opt in policy[cur_state].keys():
                cum_prob += policy[cur_state][action_opt]
                if cum_prob >= rand:
                    action = action_opt
                    break
            actions.append(action)
            rewards.append(self.reward(cur_state, action))
            next_state = self.next_state(cur_state, action)
            if next_state in self.t_states:
                terminate = True
            else:
                states.append(next_state)
        return (states, actions, rewards)

    def evaluate_MC_episode_v(self, path: Tuple[List[S], List[A], List[float]], alpha: float) -> None:
        
        states, actions, rewards = path
        state_counter = {}
        for state in states:
            state_counter[state] = 0
        g_list = [0] * len(states)
        g = 0
        for i in reversed(range(len(states))):
            g *= self.gamma
            g += rewards[i]
            g_list[i] = g
        for i in range(len(states)):
            state_counter[states[i]] += 1
            if alpha == 0:
                learning_rate = (1 / state_counter[states[i]])
            else:
                learning_rate = alpha
            self.vf[states[i]] = self.vf[states[i]] + learning_rate * (g_list[i] - self.vf[states[i]])

    def evaluate_TD_v(self, initial: S, policy: Mapping[S, Mapping[A, float]], length_param: int, iters: int, alpha: float) -> None:
        cur_state = initial
        for i in range(iters):
            update_state = cur_state
            target = 0

            #add appropriate number of rewards to target
            for j in range(length_param + 1):
                rand = np.random.uniform()
                cum_prob = 0
                action = None
                for action_opt in policy[cur_state].keys():
                    cum_prob += policy[cur_state][action_opt]
                    if cum_prob >= rand:
                        action = action_opt
                        break
                target += (self.gamma ** j) * self.reward(cur_state, action)
                cur_state = self.next_state(cur_state, action)
                if cur_state in self.t_states:
                    target += self.vf[cur_state] * (self.gamma ** (j + 1))
                    break
                
            
            #add next state value function to target
            if cur_state not in self.t_states:
                try:
                    target += self.vf[cur_state] * (self.gamma ** (length_param + 1))
                except:
                    print(cur_state)
            # update the value function
            self.vf[update_state] = self.vf[update_state] + alpha * (target - self.vf[update_state])

            while cur_state in self.t_states:
                cur_state = random.choice(list(self.s_a.keys()))

    def evaluate_TD_lambda_episode_v(self, path: Tuple[List[S], List[A], List[float]], alpha: float, lamb: float) -> None:
        states, actions, rewards = path
        for i in range(len(states)):
            g_list = [0] * (len(states) - i)
            g = 0
            for j in range(i, len(states) - 1):
                g += rewards[j] * (self.gamma ** (j-i))
                g_list[j - i] = g + self.vf[states[j + 1]] * (self.gamma ** (j-i+1))
            g_lamb = 0
            for k in range(len(g_list)):
                g_lamb += g_list[k] * (lamb ** (k - 1))
            g_lamb *= (1 - lamb)
            self.vf[states[i]] = self.vf[states[i]] + alpha * (g_lamb - self.vf[states[i]])

    def evaluate_TD_lambda_backward_v(self, initial: S, policy: Mapping[S, Mapping[A, float]], length_param: int, iters: int, alpha: float, lamb: float) -> None:
        e_trace = {}
        for state in self.s_a.keys():
            e_trace[state] = 0

        cur_state = initial
        for i in range(iters):

            update_state = cur_state
            target = 0

            #add appropriate number of rewards to target
            for j in range(length_param + 1):
                rand = np.random.uniform()
                cum_prob = 0
                action = None
                for action_opt in policy[cur_state].keys():
                    cum_prob += policy[cur_state][action_opt]
                    if cum_prob >= rand:
                        action = action_opt
                        break
                target += (self.gamma ** j) * self.reward(cur_state, action)
                cur_state = self.next_state(cur_state, action)
                if cur_state in self.t_states:
                    target += self.vf[cur_state] * (self.gamma ** (j + 1))
                    break
                
            #add next state value function to target
            if cur_state not in self.t_states:
                try:
                    target += self.vf[cur_state] * (self.gamma ** (length_param + 1))
                except:
                    print(cur_state)

            delta = target - self.vf[update_state]

            for state in self.s_a.keys():
                # update the value function
                e_trace[state] *= self.gamma
                e_trace[state] *= lamb

                if state == update_state:
                    e_trace[state] += 1
                self.vf[state] = self.vf[state] + alpha * delta * e_trace[state]
                

            while cur_state in self.t_states:
                cur_state = random.choice(list(self.s_a.keys()))
        
    def greedy_action(self, state: S) -> A:
        max = float('-inf')
        max_action = None
        for action in self.q_vf[state]:
            if self.q_vf[state][action] > max:
                max = self.q_vf[state][action]
                max_action = action
        return max_action

    def epsilon_action(self, state: S, eps: float) -> A:
        m = len(self.q_vf[state].keys())
        prob_max = eps / m + 1 - eps
        rand = np.random.uniform()
        max_action = self.greedy_action(state)
        if prob_max > rand:
            return max_action
        else:
            for action in self.q_vf[state].keys():
                if action == max_action:
                    continue
                else:
                    prob_max += eps / m
                    if prob_max > rand:
                        return action
    
    def sarsa(self, initial: S, length_param: int, iters: int, alpha: float, eps: float) -> None:
        update_state = initial
        update_action = self.epsilon_action(update_state, eps)
        for i in range(iters):
            next_state = self.next_state(update_state, update_action)
            next_action = self.epsilon_action(next_state, eps)

            target = self.reward(update_state, update_action)

            cur_state = next_state
            cur_action = next_action
            #add appropriate number of rewards to target
            for j in range(length_param):
                if cur_state in self.t_states:
                    target += self.q_vf[cur_state][self.epsilon_action(cur_state, eps)] * (self.gamma ** (j + 1))
                    break
                target += (self.gamma ** j) * self.reward(cur_state, cur_action)
                cur_state = self.next_state(cur_state, cur_action)
                cur_action = self.epsilon_action(cur_state, eps)
                    
            # add next state value function to target
            if cur_state not in self.t_states:
                try:
                    target += self.q_vf[cur_state][cur_action] * (self.gamma ** (length_param + 1))
                except:
                    print(cur_state)

            delta = target - self.q_vf[update_state][update_action]

            self.q_vf[update_state][update_action] = self.q_vf[update_state][update_action] + alpha * delta
                
            while next_state in self.t_states:
                next_state = random.choice(list(self.s_a.keys()))
                next_action = self.epsilon_action(next_state, eps)

            update_state = next_state
            update_action = next_action

    def sarsa_lambda(self, initial: S, length_param: int, iters: int, alpha: float, eps: float, lamb: float) -> None:
        e_trace = {}
        for state in self.s_a.keys():
            e_trace[state] = {}
            for action in self.s_a[state]:
                e_trace[state][action] = 0

        update_state = initial
        update_action = self.epsilon_action(update_state, eps)
        for i in range(iters):
            next_state = self.next_state(update_state, update_action)
            next_action = self.epsilon_action(next_state, eps)

            target = self.reward(update_state, update_action)

            cur_state = next_state
            cur_action = next_action
            #add appropriate number of rewards to target
            for j in range(length_param):
                if cur_state in self.t_states:
                    target += self.q_vf[cur_state][self.epsilon_action(cur_state, eps)] * (self.gamma ** (j + 1))
                    break
                target += (self.gamma ** j) * self.reward(cur_state, cur_action)
                cur_state = self.next_state(cur_state, cur_action)
                cur_action = self.epsilon_action(cur_state, eps)
                    
            # add next state value function to target
            if cur_state not in self.t_states:
                try:
                    target += self.q_vf[cur_state][cur_action] * (self.gamma ** (length_param + 1))
                except:
                    print(cur_state)

            delta = target - self.q_vf[update_state][update_action]

            for state in self.s_a.keys():
                for action in self.s_a[state]:
                    if state == update_state and action == update_action:
                        e_trace[state][action] += 1
                    self.q_vf[state][action] = self.q_vf[state][action] + alpha * delta * e_trace[state][action]
            
            for state in self.s_a.keys():
                for action in self.s_a[state]:
                    # update the value function
                    e_trace[state][action] *= self.gamma
                    e_trace[state][action] *= lamb
                
            while next_state in self.t_states:
                next_state = random.choice(list(self.s_a.keys()))
                next_action = self.epsilon_action(next_state, eps)

            update_state = next_state
            update_action = next_action

    def q_learn(self, initial: S, length_param: int, iters: int, alpha: float, eps: float) -> None:
        update_state = initial
        update_action = self.epsilon_action(update_state, eps)
        for i in range(iters):
            next_state = self.next_state(update_state, update_action)
            next_action = self.epsilon_action(next_state, eps)
            target = self.reward(update_state, update_action)

            cur_state = next_state
            cur_action = next_action
            #add appropriate number of rewards to target
            for j in range(length_param):
                if cur_state in self.t_states:
                    target += self.q_vf[cur_state][self.greedy_action(cur_state)] * (self.gamma ** (j + 1))
                    break
                target += (self.gamma ** j) * self.reward(cur_state, cur_action)
                cur_state = self.next_state(cur_state, cur_action)
                cur_action = self.epsilon_action(cur_state, eps)
                    
            # add next state value function to target
            if cur_state not in self.t_states:
                try:
                    target += self.q_vf[cur_state][self.greedy_action(cur_state)] * (self.gamma ** (length_param + 1))
                except:
                    print(cur_state)

            delta = target - self.q_vf[update_state][update_action]

            self.q_vf[update_state][update_action] = self.q_vf[update_state][update_action] + alpha * delta
                
            while next_state in self.t_states:
                next_state = random.choice(list(self.s_a.keys()))
                next_action = self.epsilon_action(next_state, eps)

            update_state = next_state
            update_action = next_action

    def derived_pol(self):
        pol = {}
        for state in self.s_a.keys():
            pol[state] = {}
            max_val = float('-inf')
            max_action = []
            for action in self.s_a[state]:
                val = self.q_vf[state][action]
                if abs(val - max_val) < 0.0001:
                    max_action.append(action)
                elif val > max_val:
                    max_val = val
                    max_action = [action]
            for action in max_action:
                pol[state][action] = 1 / len(max_action)
        return pol