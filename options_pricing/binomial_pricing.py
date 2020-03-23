import numpy as np
from options_pricing.black_scholes import option

def binomial_price_call(stock, strike, time, interest, impl_vol, steps):
    
    #model variables
    dt = time / steps
    u = np.exp(np.sqrt(dt) * impl_vol)
    d = 1 / u
    p_up = (np.exp(interest * dt) - d) / (u - d)
    p_down = 1 - p_up

    # build the final option prices
    prices = [0] * (steps + 1)
    for j in range(steps + 1):
        prices[j] = max(0, stock * (u ** j) * (d ** (steps - j)) - strike)
    
    for i in range(steps):
        for j in range(steps - i):
            opt_val = p_up * prices[j + 1] * np.exp(-interest * dt) + p_down * prices[j] * np.exp(-interest * dt)
            exercise_val = stock * (u ** j) * (d ** (steps - i - j - 1)) - strike
            if opt_val < exercise_val:
                prices[j] = exercise_val
            else:
                prices[j] = opt_val
    
    return prices[0]

def binomial_price_put(stock, strike, time, interest, impl_vol, steps):
    
    #model variables
    dt = time / steps
    u = np.exp(np.sqrt(dt) * impl_vol)
    d = 1 / u
    p_up = (np.exp(interest * dt) - d) / (u - d)
    p_down = 1 - p_up

    # build the final option prices
    prices = [0] * (steps + 1)
    for j in range(steps + 1):
        prices[j] = max(0, strike - stock * (u ** j) * (d ** (steps - j)))
    
    for i in range(steps):
        for j in range(steps - i):
            opt_val = p_up * prices[j + 1] * np.exp(-interest * dt) + p_down * prices[j] * np.exp(-interest * dt)
            exercise_val = strike - stock * (u ** j) * (d ** (steps - i - j - 1))
            if opt_val < exercise_val:
                prices[j] = exercise_val
            else:
                prices[j] = opt_val
    
    return prices[0]

price = binomial_price_call(100, 105, 0.2, 0.03, 0.24, 1000)
call = option(100, 105, 0.2, 0.03, 0.24, "call")
put = option(100, 95, 0.2, 0.03, 0.24, "put")
p_price = binomial_price_put(100, 95, 0.2, 0.03, 0.24, 1000)
print(price)    
print(call.price())
print(p_price)
print(put.price())
# divergence in put price because of early exercise option happens more often

    