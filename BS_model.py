import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__ (self,S,K,R,T,sigma, option=None):
        self.S=S
        self.K=K
        self.R=R
        self.T=T
        self.sigma=sigma
        self.option = option if option else input("Call (C), Put (P), or both: ").strip().upper()
        self.d1, self.d2 = self.calculate_d1_d2()
    
    def calculate_d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.R + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1,d2

    def black_scholes(self):
        d1, d2 = self.calculate_d1_d2()
        call = self.S * norm.cdf(d1) - self.K * np.exp(-self.R * self.T) * norm.cdf(d2)
        put = self.K * np.exp(-self.R * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        if self.option == 'C':
            return call
        elif self.option == 'P':
            return put
        else:
            return call, put

    def black_scholes_greeks(self):
        d1, d2 = self.d1, self.d2
        call_delta = norm.cdf(d1)
        put_delta = norm.cdf(-d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        call_theta = (-self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(self.T)) - self.R * self.K * np.exp(-self.R * self.T) * norm.cdf(d2)
        put_theta = (-self.S * self.sigma * norm.pdf(-d1)) / (2 * np.sqrt(self.T)) + self.R * self.K * np.exp(-self.R * self.T) * norm.cdf(-d2)
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1)
        rho_call = self.K * self.T * np.exp(-self.R * self.T) * norm.cdf(d2)
        rho_put = -self.K * self.T * np.exp(-self.R * self.T) * norm.cdf(-d2)

        if self.option == 'C':
            return {'delta': call_delta, 'gamma': gamma,'theta': call_theta, 'vega': vega,'rho': rho_call}
        elif self.option == 'P':
            return {'delta': put_delta, 'gamma': gamma,'theta': put_theta, 'vega': vega, 'rho': rho_put}
        else:
            return {'call_delta': call_delta, 'put_delta': put_delta, 'gamma': gamma,'call_theta': call_theta, 'put_theta': put_theta, 'vega': vega,'rho_call': rho_call, 'rho_put': rho_put}

    
    

    