    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm
import pyfeng as pf
import scipy.integrate as spint

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        if self.option_prices == None:
            self.option_prices = self.price(strike, spot, texp, sigma)
        self.imvol = np.zeros(strike.shape[0])
       
        for i, t_strike in enumerate(strike):
            self.imvol[i] = bsm.bsm_impvol(self.option_prices, strike, spot)
        return self.imvol
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = int(texp//dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num+1, path_num))
        stock_price_paths[0,:] = spot
        sigma_paths = np.zeros((time_num+1, path_num))
        sigma_paths[0,:] = self.sigma
      
        for i in range(time_num):
            t_row = i+1
            z = np.random.randn(path_num)
            x = np.random.randn(path_num)
            w = self.rho*z+np.sqrt(1-self.rho**2)*x
            sigma_paths[t_row,:] = sigma_paths[i,:]*np.exp(self.vov*np.sqrt(dt)*z-0.5*self.vov**2*dt)
            stock_price_paths[t_row,:] = stock_price_paths[i,:]*np.exp(sigma_paths[i,:]*np.sqrt(dt)*w-0.5*sigma_paths[i,:]**2*dt)
        self.stock_price_paths = stock_price_paths
        self.sigma_paths = sigma_paths
        self.option_prices = np.fmax(stock_price_paths[-1,:].reshape(1,-1)-strike.reshape(-1,1),0).mean(axis=1)
        return self.option_prices
    
'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        return 0

        
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = int(texp // dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num + 1, path_num))
        stock_price_paths[0, :] = spot
        sigma_paths = np.zeros((time_num + 1, path_num))
        sigma_paths[0, :] = self.sigma
        for i in range(time_num):
            t_row = i + 1
            z = np.random.randn(path_num)
            x = np.random.randn(path_num)
            w = self.rho * z + np.sqrt(1 - self.rho ** 2) * x
            sigma_paths[t_row, :] = sigma_paths[i, :] * np.exp(self.vov * np.sqrt(dt) * z - 0.5 * self.vov ** 2 * dt)
            stock_price_paths[t_row, :] = stock_price_paths[i, :] + sigma_paths[i, :] * np.sqrt(dt) * w
        self.stock_price_paths = stock_price_paths
        self.sigma_paths = sigma_paths
        self.option_prices = np.fmax(stock_price_paths[-1, :].reshape(1, -1) - strike.reshape(-1, 1), 0).mean(axis=1)
        return self.option_prices

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0

    @staticmethod
    def bsm_formula(strikes, spots, texp, vol, intr=0.0, divr=0.0, cp_sign=1):
        div_fac = np.exp(-texp * divr)
        disc_fac = np.exp(-texp * intr)
        forwards = spots / disc_fac * div_fac
        forwards = forwards.reshape(1,-1)
        strikes = strikes.reshape(-1,1)
        if (texp <= 0):
            return disc_fac * np.fmax(cp_sign * (forwards - strikes), 0)

        # floor vol_std above a very small number
        vol_std = np.fmax(vol * np.sqrt(texp), 1e-32)
        vol_std = vol_std.reshape(1,-1)

        d1 = np.log(forwards/strikes) / vol_std + 0.5 * vol_std
        d2 = d1 - vol_std

        prices = cp_sign * disc_fac * (forwards * ss.norm.cdf(cp_sign * d1) - strikes * ss.norm.cdf(cp_sign * d2))
        return prices
    
    def price(self, strike, spot, texp, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = round(texp / dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num + 1, path_num))
        stock_price_paths[0, :] = spot
        sigma_paths = np.zeros((time_num + 1, path_num))
        sigma_paths[0, :] = self.sigma
      
        for i in range(time_num):
            t_row = i + 1
            z = np.random.randn(path_num)
            sigma_paths[t_row, :] = sigma_paths[i, :] * np.exp(self.vov * np.sqrt(dt) * z - 0.5 * self.vov ** 2 * dt)
        self.sigma_paths = sigma_paths
        self.I_T = spint.simps(sigma_paths**2, dx = dt, axis=0)
        self.stock_prices = spot * np.exp(self.rho/self.vov*(sigma_paths[-1, :]-self.sigma)-
                                          self.rho**2/2*self.I_T)
        self.sigma_bs = np.sqrt((1-self.rho**2)*self.I_T/texp)
        self.option_prices = np.mean(self.bsm_formula(strike,self.stock_prices, texp, self.sigma_bs),axis=1)
        return self.option_prices
'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        return 0

    @staticmethod
    def normal_formula(strikes, spots, texp, vol, intr=0.0, divr=0.0, cp_sign=1):
        div_fac = np.exp(-texp*divr)
        disc_fac = np.exp(-texp*intr)
        forwards = spots / disc_fac * div_fac
        strikes = strikes.reshape(-1,1)
        forwards = forwards.reshape(1,-1)

        if( texp<=0 ):
            return disc_fac * np.fmax( cp_sign*(forwards-strikes), 0 )

        # floor vol_std above a very small number
        vol_std = np.fmax(vol*np.sqrt(texp), 1e-32)
        vol_std = vol_std.reshape(1,-1)
        d = (forwards-strikes)/vol_std

        prices = disc_fac*(cp_sign*(forwards-strikes)*ss.norm.cdf(cp_sign*d)+vol_std*ss.norm.pdf(d))
        return prices

    def price(self, strike, spot,texp , cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = round(texp / dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num + 1, path_num))
        stock_price_paths[0, :] = spot
        sigma_paths = np.zeros((time_num + 1, path_num))
        sigma_paths[0, :] = self.sigma
      
        for i in range(time_num):
            t_row = i + 1
            z = np.random.randn(path_num)
            sigma_paths[t_row, :] = sigma_paths[i, :] * np.exp(self.vov * np.sqrt(dt) * z - 0.5 * self.vov ** 2 * dt)
        self.sigma_paths = sigma_paths
        self.I_T = spint.trapezoid(sigma_paths ** 2, dx=dt, axis=0)
        self.stock_prices = spot + self.rho / self.vov * (sigma_paths[-1, :] - self.sigma)
        self.sigma_n = np.sqrt((1 - self.rho ** 2) * self.I_T / texp)
        self.option_prices = np.mean(self.normal_formula(strike, self.stock_prices, texp, self.sigma_n), axis=1)
        return self.option_prices