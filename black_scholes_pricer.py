from scipy.stats import norm
import numpy as np
import copy

class Option:
    def __init__(self, S, K, r, T, sigma, q, option_type, nb_sim=10_000, nb_steps=252, seed=123):
        assert option_type.lower() in ['call', 'put'], 'option_type must be call or put'
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self.nb_sim = nb_sim
        self.nb_steps = nb_steps
        self.seed = seed
    
##########################################################################################################################################################################################################

class EuropeanOption(Option):
    pass

class AmericanOption(Option):
    pass

class DigitalOption(Option):
    def __init__(self, S, K, r, T, sigma, q, option_type, nb_sim=10_000, nb_steps=252, payout=1, seed=123):
        super().__init__(S, K, r, T, sigma, q, option_type, nb_sim, nb_steps, seed)
        self.payout = payout

class OneTouchOption(Option):
    def __init__(self, S, K, r, T, sigma, q, option_type, nb_sim=10_000, nb_steps=252, payout=1, seed=123):
        super().__init__(S, K, r, T, sigma, q, option_type, nb_sim, nb_steps, seed)
        self.payout = payout

class BarrierOption(Option):
    def __init__(self, S, K, r, T, sigma, q, option_type, H, barrier_type, direction, exercise_type, nb_sim=10_000, nb_steps=252, seed=123):
        super().__init__(S, K, r, T, sigma, q, option_type, nb_sim, nb_steps, seed)
        assert barrier_type.lower() in ['in', 'out'], 'Barrier type must be In or Out'
        assert direction.lower() in ['up', 'down'], 'Direction must be Up or Down'
        assert exercise_type.lower() in ['european', 'american'], 'Exercise type must be European or American'
        self.H = H
        self.barrier_type = barrier_type
        self.direction = direction
        self.exercise_type = exercise_type

class AsianOption(Option):
    def __init__(self, S, K, r, T, sigma, q, option_type, average_type, average_freq, asianing_type, nb_obs=0, nb_sim=100_000, nb_steps=252, seed=123):
        super().__init__(S, K, r, T, sigma, q, option_type, nb_sim, nb_steps, seed)
        assert average_type.lower() in ['arithmetic', 'geometric'], 'Averaging type must be Arithmetic or Geometric'
        assert average_freq.lower() in ['daily', 'weekly', 'monthly', 'quarterly', 'semi-annual', 'annual'], 'Averaging frequency must be Daily, Weekly, Monthly, Quarterly, Semi-annual or Annual'
        assert asianing_type.lower() in ['full', 'partial'], 'Asianing type must be Full or Partial'
        self.average_type = average_type
        self.average_freq = average_freq
        self.asianing_type = asianing_type
        self.nb_obs = nb_obs

##########################################################################################################################################################################################################
    
class NumericalGreeks:
    """
        ng = NumericalGreeks(option, pricing_fn)
        ng.greeks()   # -> dict

    pricing_function must be a callable: f(option) -> float
    """
    def __init__(self, option, pricing_function):
        self.option = option
        self.pricing_function = pricing_function

    def greeks(self):
        option = self.option
        h_S = option.S * 0.01
        h_v = 0.01
        h_t = 1 / 365
        h_r = 0.0001

        price_base = self.pricing_function(option)

        def bump(**kwargs):
            o = copy.copy(option) # shallow copy of the option, avoids modifying the original
            for k, v in kwargs.items(): # overwrite only the bumped parameter(s)
                setattr(o, k, v)
            return self.pricing_function(o) # reprice with the bumped parameter

        up_S  = bump(S=option.S + h_S)
        dn_S  = bump(S=option.S - h_S)
        delta = (up_S - dn_S) / (2 * h_S)
        gamma = (up_S - 2 * price_base + dn_S) / h_S**2

        up_v  = bump(sigma=option.sigma + h_v)
        dn_v  = bump(sigma=option.sigma - h_v)
        vega  = (up_v - dn_v) / (2 * h_v) / 100
        volga = (up_v - 2 * price_base + dn_v) / h_v**2

        theta = (bump(T=option.T - h_t) - price_base) / h_t / 365

        rho = (bump(r=option.r + h_r) - bump(r=option.r - h_r)) / (2 * h_r) / 100

        vanna = (
            bump(S=option.S + h_S, sigma=option.sigma + h_v)
          - bump(S=option.S + h_S, sigma=option.sigma - h_v)
          - bump(S=option.S - h_S, sigma=option.sigma + h_v)
          + bump(S=option.S - h_S, sigma=option.sigma - h_v) 
        ) / (4 * h_S * h_v)

        delta_t1 = (
            bump(S=option.S + h_S, T=option.T - h_t)
          - bump(S=option.S - h_S, T=option.T - h_t)
        ) / (2 * h_S)
        charm = (delta_t1 - delta) / h_t

        return {
            "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta,
            "Rho": rho, "Volga": volga, "Vanna": vanna, "Charm": charm,
        }
    
##########################################################################################################################################################################################################

class BS_Analytical:

    def _d1_d2(self, option: Option) -> float:
        d1 = (np.log(option.S / option.K) + (option.r - option.q + 0.5 * option.sigma**2) * option.T) / (option.sigma * np.sqrt(option.T))        
        d2 = d1 - option.sigma * np.sqrt(option.T)
        return d1, d2

    def eu_vanilla_option_price(self, option: EuropeanOption) -> float:
        d1, d2 = self._d1_d2(option)

        if option.option_type == 'call':
            price = (np.exp(-option.q * option.T) * option.S * norm.cdf(d1) - 
                     np.exp(-option.r * option.T) * option.K * norm.cdf(d2))
        else:
            price = (np.exp(-option.r * option.T) * option.K * norm.cdf(-d2) - 
                     np.exp(-option.q * option.T) * option.S * norm.cdf(-d1))
        return price
    
    def eu_vanilla_option_greeks(self, option: EuropeanOption) -> dict:
        d1, d2 = self._d1_d2(option)

        gamma = norm.pdf(d1) / (option.S * option.sigma * np.sqrt(option.T))

        vega = norm.pdf(d1) * option.S * np.sqrt(option.T)

        vanna = -norm.pdf(d1) * d2 / option.sigma

        volga = option.S * norm.pdf(d1) * np.sqrt(option.T) * d1 * d2 / option.sigma

        if option.option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (option.S * norm.pdf(d1) * option.sigma) / (2 * np.sqrt(option.T))
                    - option.r * option.K * np.exp(-option.r * option.T) * norm.cdf(d2)) / 365
            rho = option.K * option.T * np.exp(-option.r * option.T) * norm.cdf(d2) / 100
            charm = -norm.pdf(d1) * (2 * option.r * option.T - d2 * option.sigma * np.sqrt(option.T)) / (2 * option.T * option.sigma * np.sqrt(option.T))
        else:
            delta = -norm.cdf(-d1)
            theta = (- (option.S * norm.pdf(d1) * option.sigma) / (2 * np.sqrt(option.T))
                    + option.r * option.K * np.exp(-option.r * option.T) * norm.cdf(-d2)) / 365
            rho = -option.K * option.T * np.exp(-option.r * option.T) * norm.cdf(-d2) / 100
            charm = -norm.pdf(d1) * (2 * option.r * option.T - d2 * option.sigma * np.sqrt(option.T)) / (2 * option.T * option.sigma * np.sqrt(option.T)) + option.r * np.exp(-option.r * option.T) * norm.cdf(-d1) 
            
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
    
    def digital_option_price(self, option: DigitalOption) -> float:
        _, d2 = self._d1_d2(option)
        return (option.payout * np.exp(-option.r * option.T) * norm.cdf(d2) if option.option_type == 'call' 
                else option.payout * np.exp(option.r * option.T) * norm.cdf(-d2))
    
    def digital_option_greeks(self, option: DigitalOption) -> dict:
        d1, d2 = self._d1_d2(option)
        disc   = np.exp(-option.r * option.T)
        
        if option.option_type == 'call':
            delta = option.payout * disc * norm.pdf(d2) / (option.S * option.sigma * np.sqrt(option.T))
            gamma = -option.payout * disc * norm.pdf(d2) * d1 / (option.S**2 * option.sigma**2 * option.T)
            vega  = -option.payout * disc * norm.pdf(d2) * d1 / (option.sigma * 100)
            theta = (option.payout * disc * norm.pdf(d2) * d1 / (2 * option.T) + 
                    option.r * disc * norm.pdf(d2)) / 365
            rho   = option.payout * (-option.T * disc * norm.cdf(d2) + 
                    disc * norm.pdf(d2) / (option.sigma * np.sqrt(option.T))) / 100
        else:
            delta = -option.payout * disc * norm.pdf(d2) / (option.S * option.sigma * np.sqrt(option.T))
            gamma = option.payout * disc * norm.pdf(d2) * d1 / (option.S**2 * option.sigma**2 * option.T)
            vega  = option.payout * disc * norm.pdf(d2) * d1 / (option.sigma * 100)
            theta = (-option.payout * disc * norm.pdf(d2) * d1 / (2 * option.T) + 
                    option.r * disc * norm.pdf(-d2)) / 365
            rho   = option.payout * (option.T * disc * norm.cdf(-d2) - 
                    disc * norm.pdf(d2) / (option.sigma * np.sqrt(option.T))) / 100
            
        greeks = {
            'Delta': delta, 
            'Gamma': gamma, 
            'Vega': vega, 
            'Theta': theta, 
            'Rho': rho
        } 

        return greeks
    
##########################################################################################################################################################################################################

class BS_Binomial:

    def vanilla_option_price(self, option: AmericanOption) -> float:
        dt = option.T / option.nb_steps
        u = np.exp(option.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((option.r - option.q) * dt) - d) / (u - d)
        discount = np.exp(-option.r * dt)

        final_prices = np.array([option.S * (u ** (self.steps - 2 * j)) for j in range(self.steps + 1)])

        if option.option_type == 'call':
            payoff = np.maximum(final_prices - option.K, 0)
        else:
            payoff = np.maximum(option.K - final_prices, 0)

        for i in range(self.steps - 1, -1, -1):
            payoff = discount * (p * payoff[:i + 1] + (1 - p) * payoff[1:i + 2])
            node_prices = np.array([option.S * (u ** (i - 2 * j)) for j in range(i + 1)])
            if option.option_type == 'call':
                payoff = np.maximum(node_prices - option.K, payoff)
            else:
                payoff = np.maximum(option.K - node_prices, payoff)
        
        return payoff[0]
    
##########################################################################################################################################################################################################

    
class BS_MonteCarlo:

    def _simulate_terminal_price(self, option):
        np.random.seed(option.seed)
        z = np.random.standard_normal(option.nb_sim)
        return option.S * np.exp((option.r - option.q - 0.5 * option.sigma**2) * option.T + option.sigma * np.sqrt(option.T) * z)
    
    def _simulate_paths(self, option):
        np.random.seed(option.seed)
        dt = option.T / option.nb_steps
        drift = (option.r - option.q - 0.5 * option.sigma**2) * dt
        diffusion = option.sigma * np.sqrt(dt)

        z = np.random.standard_normal((option.nb_sim, option.nb_steps))          
        log_returns = drift + diffusion * z                         
        log_prices  = np.log(option.S) + np.cumsum(log_returns, axis=1)  
        paths = np.exp(log_prices)
        return paths
    
    def eu_vanilla_option_price(self, option: EuropeanOption) -> float:
        S_T = self._simulate_terminal_price(option)

        if option.option_type == 'call':
            payoffs = np.maximum(S_T - option.K, 0)
        else:
            payoffs = np.maximum(option.K - S_T, 0)
        
        return np.exp(-option.r * option.T) * np.mean(payoffs)
    
    def digital_option_price(self, option: DigitalOption) -> float:
        S_T = self._simulate_terminal_price(option)

        if option.option_type == 'call':
            breached = S_T >= option.K
        else:
            breached = S_T <= option.K

        return np.exp(-option.r * option.T) * np.mean(breached)

    def one_touch_option_price(self, option: OneTouchOption):
        paths = self._simulate_paths(option)

        if option.option_type == 'call':
            touched = np.any(paths >= option.K, axis=1)            
        else:
            touched = np.any(paths <= option.K, axis=1)            

        return np.exp(-option.r * option.T) * np.mean(touched)

    def barrier_option_price(self, option: BarrierOption):
        if option.exercise_type == 'european':
            S_T = self._simulate_terminal_price(option)

            if option.direction == 'up':
                hit = S_T >= option.H
            else:
                hit = S_T <= option.H

        else:
            paths = self._simulate_paths(option)
            S_T = paths[:, -1]  

            if option.direction == 'up':
                hit = np.any(paths >= option.H, axis=1)
            else:
                hit = np.any(paths <= option.H, axis=1)

        valid = hit if option.barrier_type == 'in' else ~hit

        if option.option_type == 'call':
            payoff = np.maximum(S_T - option.K, 0)
        else:
            payoff = np.maximum(option.K - S_T, 0)

        payoff *= valid.astype(float)
        return np.exp(-option.r * option.T) * np.mean(payoff)

    def asian_option_price(self, option: AsianOption):
        freq_map = {
            'annual': 1, 
            'semi-annual': 2,
            'quarterly': 4,
            'monthly': 12,
            'weekly': 52,
            'daily': 252
        }

        f = option.average_freq.lower()
        m = freq_map[f]
        
        if option.asianing_type == 'partial' and option.nb_obs >= m:
            raise ValueError('The number of observations should be lower than the total number of observations! Otherwise use Full Asianing') 

        np.random.seed(option.seed)
        dt = option.T / m
        drift = (option.r - option.q - 0.5 * option.sigma**2) * dt
        diffusion = option.sigma * np.sqrt(dt)

        z = np.random.standard_normal((option.nb_sim, m))
        log_returns = drift + diffusion * z
        log_prices = np.log(option.S) + np.cumsum(log_returns, axis=1)
        paths = np.exp(log_prices)

        if option.average_type == 'arithmetic':
            if option.asianing_type == 'full':
                avg = paths.mean(axis=1)
            else:
                paths = paths[:, m-option.nb_obs:]
                avg = paths.mean(axis=1)
        else:
            if option.asianing_type == 'full':
                avg = np.exp(log_prices.mean(axis=1))
            else:
                log_prices = log_prices[:, m-option.nb_obs:]
                avg = np.exp(log_prices.mean(axis=1)) 

        if option.option_type == 'call':
            payoff = np.maximum(avg - option.K, 0)
        else:
            payoff = np.maximum(option.K - avg, 0)

        return np.exp(-option.r * option.T) * np.mean(payoff)
    
if __name__ == '__main__':
    mc = BS_MonteCarlo()
    an = BS_Analytical()

    digital = DigitalOption(S=100, K=120, T=1, r=0.02, sigma=0.20, q=0.035, option_type='call')
    one_touch = OneTouchOption(S=100, K=120, T=1, r=0.02, sigma=0.20, q=0.035, option_type='call')
    barrier_option = BarrierOption(S=100, K=100, T=1, r=0.02, sigma=0.20, q=0.035, option_type='call', H=120, barrier_type='out', direction='up', exercise_type='american')
    asian_option = AsianOption(S=100, K=120, T=1, r=0.02, sigma=0.20, q=0.035, option_type='call', average_type='arithmetic', average_freq='monthly', asianing_type='full', nb_obs=3)

    price = mc.asian_option_price(asian_option)
    print(price)