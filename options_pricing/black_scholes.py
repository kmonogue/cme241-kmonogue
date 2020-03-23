from scipy.stats import norm
import numpy as np

class option:

    def __init__(self, stock_, strike_, time_, interest_, impl_vol_, type_):
        self.stock = stock_
        self.strike = strike_
        self.time = time_
        self.interest = interest_
        self.impl_vol = impl_vol_
        self.type = type_
        self.d1 = (1 / (impl_vol_ * np.sqrt(time_))) * (np.log(stock_ / strike_) + (interest_ + (impl_vol_ ** 2) / 2) * time_)
        self.d2 = self.d1 - impl_vol_ * np.sqrt(time_)

    def price(self):
        call_price = norm.cdf(self.d1) * self.stock - norm.cdf(self.d2) * self.strike * np.exp(-self.interest * self.time)
        if self.type == "call":
            return call_price
        else:
            return self.strike * np.exp(-self.interest * self.time) - self.stock + call_price

    def delta(self):
        if self.type == "call":
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.stock * self.impl_vol * np.sqrt(self.time))

    def vega(self):
        return self.stock * norm.pdf(self.d1) * np.sqrt(self.time) / 100

    def theta(self):
        part = (-self.stock * norm.pdf(self.d1) * self.impl_vol) / (2 * np.sqrt(self.time))
        part2 = self.interest * self.strike * np.exp(-self.interest * self.time)
        if self.type == "call":
            return (part - part2 * norm.cdf(self.d2)) / 365
        else:
            return (part + part2 * norm.cdf(-self.d2)) / 365

    def rho(self):
        part = -self.strike * self.time * np.exp(self.time * -self.interest)
        if self.type == "call":
            return part * norm.cdf(self.d2) / 10000
        else:
            return -part * norm.cdf(-self.d2) / 10000


if __name__ == '__main__':
    call = option(100, 105, 0.2, 0.03, 0.24, "call")
    put = option(100, 105, 0.2, 0.03, 0.24, "put")

    print(call.price())
    print(call.delta())
    print(call.gamma())
    print(call.theta())
    print(call.vega())
    print(call.rho())
    print()
    print(put.price())
    print(put.delta())
    print(put.gamma())
    print(put.theta())
    print(put.vega())
    print(put.rho())
