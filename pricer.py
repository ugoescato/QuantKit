from scipy.stats import norm
import numpy as np

class Option:
    def __init__(self, S, K, r, T, sigma, q, option_type):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.q = q
        self.option_type = option_type
        assert self.option_type in ['call', 'put'], 'Option type must be call or put'

class Analytical(Option):
    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))        
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def price(self):
        d1, d2 = self._d1_d2()

        if self.option_type == 'call':
            price = (np.exp(-self.q * self.T) * self.S * norm.cdf(d1) - 
                     np.exp(-self.r * self.T) * self.K * norm.cdf(d2))
        else:
            price = (np.exp(-self.r * self.T) * self.K * norm.cdf(-d2) - 
                     np.exp(-self.q * self.T) * self.S * norm.cdf(-d1))
        return price
    
    def greeks(self):
        d1, d2 = self._d1_d2()

        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

        vega = norm.pdf(d1) * self.S * np.sqrt(self.T)

        vanna = -norm.pdf(d1) * d2 / self.sigma

        volga = self.S * norm.pdf(d1) * np.sqrt(self.T) * d1 * d2 / self.sigma

        if self.option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) / 365
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
            charm = -norm.pdf(d1) * (2 * self.r * self.T - d2 * self.sigma * np.sqrt(self.T)) / (2 * self.T * self.sigma * np.sqrt(self.T))
        else:
            delta = -norm.cdf(-d1)
            theta = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 365
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
            charm = -norm.pdf(d1) * (2 * self.r * self.T - d2 * self.sigma * np.sqrt(self.T)) / (2 * self.T * self.sigma * np.sqrt(self.T)) + self.r * np.exp(-self.r * self.T) * norm.cdf(-d1) 
            
        greeks = {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho,
            'Vanna': vanna,
            'Volga': volga,
            'Charm': charm
        }

        return greeks
    
if __name__ == '__main__':
    method = Analytical(S=100, K=100, r=0, T=1, sigma=0.2, q=0, option_type='call')    
    print(method.price())
    print(method.greeks())