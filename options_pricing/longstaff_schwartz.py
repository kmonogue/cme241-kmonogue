import numpy as np

# TYPES
# sp = 2d np array of stock paths
# rf = float
# dt = float
# feature_funcs = list of functions
# strike = float/int
def longstaff_schwartz_call(sp, rf, dt, feature_funcs, strike):
    
    num_paths = sp.shape[0]
    num_steps = sp.shape[1]

    # fill the initial (final) values
    cf = np.zeros(num_paths)
    for i in range(num_paths):
        cf[i] = max(0, sp[i][-1] - strike)

    print(cf)
    
    #perform algorithm
    x = np.zeros((num_paths, len(feature_funcs)))
    y = np.zeros(num_paths)
    for j in reversed(range(1, num_steps)):
        cf = cf * np.exp(-rf * dt)
        for i in range(num_paths):
            price = sp[i][j]
            payoff = max(0, price - strike)
            if payoff > 0:
                y[i] = cf[i]
                for k, func in enumerate(feature_funcs):
                    x[i][k] = func(sp[i][:j+1])
        w = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        for i in range(num_paths):
            payoff = sp[i][j]
            payoff = max(0, price - strike)
            if payoff > np.dot(w.T, x[i]):
                cf[i] = payoff
    exercise = max(sp[0][0] - strike, 0)
    val = np.exp(-rf * dt) * np.mean(cf)
    return max(val, exercise)

def generate_returns(initial_price, num_paths, num_steps, mean, var):
    sp = np.zeros((num_paths, num_steps))
    for i in range(num_steps):
        sp[i][0] = initial_price
        returns = np.random.normal(mean, var, num_steps)
        print(returns)
        for j in range(1, num_steps):
            sp[i][j] = sp[i][j-1] * (1 + returns[j])
    return sp

class mean_feature:
    def __call__(self, path):
        return np.mean(path)

class std_feature:
    def __call__(self, path):
        return np.std(path)

class rolling_feature:
    def __init__(self, length):
        self.length = length

    def __call__(self, path):
        window = max(path.size, self.length)
        if window == 1:
            return 0
        else:
            return np.mean(path[-window:])

feature_funcs = [mean_feature(), std_feature()]

sp = generate_returns(300, 10, 10, 0.005, 0.01)
val = longstaff_schwartz_call(sp, 0.01, 0.08, feature_funcs, 300)
print(val)
    
    