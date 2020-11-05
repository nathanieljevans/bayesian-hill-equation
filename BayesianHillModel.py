'''
# TODO: create conda env 
# TODO: remove unnecessary libraries
# TODO: add get_AUC function
'''
import scipy.special as sps
import pyro 
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from pyro.infer import MCMC, NUTS
from scipy.stats import norm
from torch import nn
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroSample
from pyro.infer import Predictive

pyro.enable_validation(True)
pyro.set_rng_seed(1)

import os

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import statsmodels.api as sm
import statsmodels

from scipy.stats import norm, gamma, poisson, beta 
from  pyro.infer.mcmc.util import summary


def gamma_modes_to_params(E, S): 
    '''
    E ~ mean 
    S ~ variance 

    mean, var -> alpha, beta for shape, rate parameterization of the gamma distribution 

    return alpha, beta 
    '''
    beta = E/S 
    alpha = E**2/S 
    
    return alpha, beta

class BayesianHillModel(): 
    def __init__(self): 
        '''
        '''
        self.prior = dict()

    def get_priors(self): 
        '''
        Default Prior values stored here. 
        '''
        E0_mean = 1.
        E0_std = self.prior.get('E0_std', 0.1)

        alpha_emax, beta_emax = gamma_modes_to_params(self.prior.get('Emax_Mean', 0.5), self.prior.get('Emax_var', 1.))

        alpha_H, beta_H = gamma_modes_to_params(self.prior.get('H_Mean', 1.5), self.prior.get('H_var', 1.))

        log10_ec50_mean = self.prior.get('log10_EC50_Mean', -3)
        log10_ec50_std = self.prior.get('log10_EC50_Var', 2.)

        alpha_obs, beta_obs = gamma_modes_to_params(self.prior.get('Obs_std_Mean', 1.), self.prior.get('Obs_std_Var', 1.))


        return E0_mean, E0_std, alpha_emax, beta_emax, alpha_H, beta_H, log10_ec50_mean, log10_ec50_std, alpha_obs, beta_obs

    def plot_priors(self): 
        '''
        '''
        E0_mean, E0_std, alpha_emax, beta_emax, alpha_H, beta_H, log10_ec50_mean, log10_ec50_std, alpha_obs, beta_obs = self.get_priors()

        f, axes = plt.subplots(2,3, figsize=(12,7))

        # E0 
        xx = np.linspace(0, 2, 50)
        rv = norm(E0_mean, E0_std)
        yy = rv.pdf(xx)
        axes.flat[0].set_title('E0 parameter')
        axes.flat[0].set_xlabel('E0')
        axes.flat[0].set_ylabel('probability')
        axes.flat[0].plot(xx, yy, 'r-')

        # EMAX 
        xx = np.linspace(0, 2, 50)
        rv = gamma(alpha_emax, scale=1/beta_emax, loc=0)
        yy = rv.pdf(xx)
        axes.flat[1].set_title('Emax parameter')
        axes.flat[1].set_xlabel('Emax')
        axes.flat[1].set_ylabel('probability')
        axes.flat[1].plot(xx, yy, 'r-')

        # H 
        xx = np.linspace(0, 5, 100)
        rv = gamma(alpha_H, scale=1/beta_H, loc=0)
        yy = rv.pdf(xx)
        axes.flat[2].set_title('Hill Coefficient (H) parameter')
        axes.flat[2].set_xlabel('H')
        axes.flat[2].set_ylabel('probability')
        axes.flat[2].plot(xx, yy, 'r-')

        # EC50 
        xx = np.logspace(-7,1, 100)
        rv = norm(log10_ec50_mean, log10_ec50_std)
        yy = rv.pdf(np.log10(xx))
        axes.flat[3].set_title('EC50 parameter')
        axes.flat[3].set_xlabel('EC50 [uM]')
        axes.flat[3].set_ylabel('probability')
        axes.flat[3].plot(xx, yy, 'r-')

        # Log10 EC50 
        axes.flat[4].set_title('Log10 EC50 parameter [~ Normal]')
        axes.flat[4].set_xlabel('Log10( EC50 [uM] )')
        axes.flat[4].set_ylabel('probability')
        axes.flat[4].plot(np.log10(xx), yy, 'r-')

        # OBS 
        xx = np.linspace(0, 5, 100)
        rv = gamma(alpha_obs, scale=1/beta_obs, loc=0)
        yy = rv.pdf(xx)
        axes.flat[5].set_title('Observation Std parameter')
        axes.flat[5].set_xlabel('Obs. Std')
        axes.flat[5].set_ylabel('probability')
        axes.flat[5].plot(xx, yy, 'r-')

        plt.tight_layout()
        plt.show()

    def plot_prior_regression(self, n_samples=1000, savepath=None, verbose=True): 
        '''
        '''
        XX = torch.tensor( np.logspace(-9, 4, 100) )

        samples = [self.model(XX) for i in range(n_samples)]

        plt.figure(figsize=(7,7))

        for i,s in enumerate(samples):
            if (i%100==0) and verbose: print(f'plotting prior regression...{i/n_samples*100:.1f}%', end='\r') 
            plt.plot(np.log10(XX), s, 'r-', alpha=0.005, linewidth=4.0)
            
        plt.xlabel('log10 Concentration')
        plt.ylabel('cell_viability')
        plt.ylim(0,1.2)
        plt.legend()
        plt.title('Prior Probability Hill Regression')

        if savepath is not None: 
            plt.savefig(savepath + '/prior_regressions.png')
        else: 
            plt.show()


    def model(self, X, Y=None):
        '''
        
        '''
        E0_mean, E0_std, alpha_emax, beta_emax, alpha_H, beta_H, log10_ec50_mean, log10_ec50_std, alpha_obs, beta_obs = self.get_priors()

        E0 = pyro.sample('E0', dist.Normal(E0_mean, E0_std))

        Emax = pyro.sample('Emax', dist.Beta(alpha_emax, beta_emax))
        
        H = pyro.sample('H', dist.Gamma(alpha_H, beta_H))
        
        EC50 = 10**pyro.sample('log_EC50', dist.Normal(log10_ec50_mean, log10_ec50_std))

        obs_sigma = pyro.sample("obs_sigma", dist.Gamma(alpha_obs, beta_obs))
        
        obs_mean = E0 + (Emax - E0)/(1+(EC50/X)**H)

        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(obs_mean.squeeze(-1), obs_sigma), obs=Y)
            
        return obs_mean

    def check_converged(self, Rhat_tol=0.05, verbose=False): 
        '''
        '''
        results = self.summary()
        max_rhat = max(results.loc['r_hat',:].values); min_rhat=min(results.loc['r_hat',:].values)
        if verbose: print('max/min rhat:', (max_rhat, min_rhat))
        return ~(max_rhat > (1+Rhat_tol) or min_rhat < (1-Rhat_tol))


    def fit(self, X, Y, num_samples=500, burnin=150, num_chains=1, seed=1): 
        '''
        '''
        self.X = X 
        self.Y = Y

        if seed is not None: 
            torch.manual_seed(seed)

        nuts_kernel = NUTS(self.model, adapt_step_size=True)
        self.mcmc_res = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin, num_chains=num_chains)
        self.mcmc_res.run(X,Y)

    def plot_fitted_params(self, savepath=None): 
        samples = {k: v.detach().cpu().numpy() for k, v in self.mcmc_res.get_samples().items()}

        f, axes = plt.subplots(3,2, figsize=(10,5))

        for ax, key in zip(axes.flat, samples.keys()): 
            
            ax.set_title(key)
            ax.hist(samples[key], bins=np.linspace(min(samples[key]), max(samples[key]), 50), density=True)
            ax.set_xlabel(key)
            ax.set_ylabel('probability')
            
        axes.flat[-1].hist(10**samples['log_EC50'], bins=np.linspace(min(10**(samples['log_EC50'])), max(10**(samples['log_EC50'])), 50))
        axes.flat[-1].set_title('EC50')
        axes.flat[-1].set_xlabel('EC50 [uM]')
            
        plt.tight_layout()
        
        if savepath is not None: 
            plt.savefig(savepath)
        else: 
            plt.show()

    def plot_fit(self, savepath=None): 
        samples = {k: v.detach().cpu().numpy() for k, v in self.mcmc_res.get_samples().items()}

        plt.figure(figsize=(7,7))

        xx = np.logspace(-7, 6, 200)

        for i,s in pd.DataFrame(samples).iterrows(): 
            yy = s.E0 + (s.Emax - s.E0)/(1+(10**s.log_EC50/xx)**s.H)
            plt.plot(np.log10(xx), yy, 'ro', alpha=0.01)
            
            
        plt.plot(np.log10(self.X), self.Y, 'b.', label='data')
        plt.xlabel('log10 Concentration')
        plt.ylabel('cell_viability')
        plt.ylim(0,1.2)
        plt.legend()
        plt.title('MCMC results')
        
        if savepath is not None: 
            plt.savefig(savepath)
        else: 
            plt.show()

    def summary(self, p=0.9, verbose=False): 
        self.results = pd.DataFrame( summary(self.mcmc_res._samples, prob=0.9) )
        if verbose: print(self.results)
        return self.results

    def get_samples(self): 
        return pd.DataFrame({key: np.array(self.mcmc_res._samples[key]).flatten() for key in self.mcmc_res._samples})
    
    def get_ICxx(self, xx): 
        # returned in log concentration
        ics = []
        for i,row in self.get_samples().iterrows(): 
            try: 
                ic = np.exp( (1/row.H)*(np.log((row.Emax - row.E0)/(xx - row.E0))) - np.log(10**row.log_EC50) )
                ics.append(ic)
            except: 
                ics.append(np.inf)
        return ics
                          
    

if __name__ == '__main__':
    
    print('running test example...')
    Y = torch.tensor([1., 1., 1., 0.9, 0.7, 0.6, 0.5], dtype=torch.float)
    X = torch.tensor([10./3**i for i in range(7)][::-1], dtype=torch.float).unsqueeze(-1)

    Model = BayesianHillModel()

    Model.plot_priors()
    Model.plot_prior_regression(n_samples=10000)
    Model.fit(X,Y, num_samples=400, burnin=100)
    Model.plot_fitted_params()
    Model.plot_fit()
    Model.summary(verbose=True)
    conv = Model.check_converged(verbose=True); print(conv)

    