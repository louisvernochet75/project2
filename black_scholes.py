import numpy as np
from scipy.stats import norm
import pandas as pd 


class BlackScholes:
    def __init__ (self,S,K,R,T,sigma, option=None):
        self.S=float(S)
        self.K=float(K)
        self.R=float(R)
        self.T=float(T)
        self.sigma=float(sigma)
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
            return round(call,4)
        elif self.option == 'P':
            return round(put,4)
        else:
            return round(call, 4), round(put, 4)
        
    
    @staticmethod
    def batch_compute(stocks_info, R, T, option='BOTH'):
        """
        stocks_info : list[dict] ou DataFrame avec colonnes 'Stock', 'S', 'K', 'sigma'
        """
        results = []
        greeks_data = []

        for info in stocks_info:
            stock = info['Stock']
            S = info['S']
            K = info['K']
            sigma = info['sigma']

            bs = BlackScholes(S, K, R, T, sigma, option)
            price = bs.black_scholes()
            greeks = bs.black_scholes_greeks()
            greeks['Stock'] = stock
            greeks_data.append(greeks)

            if option == 'BOTH':
                call_price, put_price = price
            elif option == 'C':
                call_price, put_price = price, None
            else:
                call_price, put_price = None, price

            results.append({
                'Stock': stock,
                'Spot (€)': round(S, 2),
                'Strike (€)': round(K, 2),
                'Call Price (€)': call_price,
                'Put Price (€)': put_price
            })

        return pd.DataFrame(results), pd.concat(greeks_data, ignore_index=True)



    def black_scholes_greeks(self):
        d1, d2 = self.d1, self.d2
        call_delta = norm.cdf(d1)
        put_delta = norm.cdf(-d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        call_theta = (-self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(self.T)) - self.R * self.K * np.exp(-self.R * self.T) * norm.cdf(d2)
        put_theta = (-self.S * self.sigma * norm.pdf(-d1)) / (2 * np.sqrt(self.T)) + self.R * self.K * np.exp(-self.R * self.T) * norm.cdf(-d2)
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1)
        call_rho = self.K * self.T * np.exp(-self.R * self.T) * norm.cdf(d2)
        put_rho = -self.K * self.T * np.exp(-self.R * self.T) * norm.cdf(-d2)

        def format_floats(vals):
            return [round(v, 4) for v in vals]

        if self.option == 'C':
            result = {'Greek': ['delta', 'gamma', 'theta', 'vega', 'rho'],
                      'Value': [call_delta, gamma, call_theta, vega, call_rho]}
            return pd.DataFrame(result)
        elif self.option == 'P':
            result = {'Greek': ['delta', 'gamma', 'theta', 'vega', 'rho'],
                      'Value': [put_delta, gamma, put_theta, vega, put_rho]}
            return  pd.DataFrame(result)
        else:
            result={'Greek':['Delta','Gamma','Theta','Vega','Rho'],
                    'Call':[call_delta, gamma, call_theta,vega,call_rho],
                    'Put':[put_delta,gamma,put_theta,vega,put_rho]}
            return pd.DataFrame(result)

