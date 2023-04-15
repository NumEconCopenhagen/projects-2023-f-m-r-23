
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0
        # par.dummy = 0.0
        par.mu = 0.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF,sigma=None,alpha=None):
        
        """ calculate utility """
        
        par = self.par
        sol = self.sol
        
        sigma = par.sigma if sigma is None else sigma
        alpha = par.alpha if alpha is None else alpha

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production, adjusted for different values of sigma
        if sigma==1:
            H = HM**(1-alpha)*HF**alpha
        elif sigma==0:
            H = np.min(HM,HF)
        else:
            H = ((1-alpha)*HM**((sigma-1)/sigma)+alpha*HF**((sigma-1)/sigma))**(sigma/(sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work (with addition of dummy (Question 5))
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_+ (LF)**par.mu-1) 
        
        return utility - disutility

    def solve_discrete(self,sigma=None,alpha=None, do_print=False):
       
        """ solve model discretely """
        
        # setting up initial parameters
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # accounting for None values of Sigma and Alpha
        sigma = par.sigma if sigma is None else sigma
        alpha = par.alpha if alpha is None else alpha
        
        # all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # calculate utility
        u = self.calc_utility(LM,HM,LF,HF,sigma,alpha)
    
        # set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_continuous(self,do_print=False):
        """ solve model continuously """
        
        # setting up initial parameters
        par = self.par
        sol = self.par
        opt = SimpleNamespace()
        
        # setting bounds, 24 hours for each 
        bnds = ((0,24),(0,24),(0,24),(0,24))
        
        # setting up constraints, 24 hour max for H and L
        cnst = {'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1],
                'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]}

        # creating objective function for optimize
        def obj(x):
            return -self.calc_utility(x[0],x[1],x[2],x[3])

        #Optimizing
        res = optimize.minimize(obj,x0 = (4.5,4.5,4.5,4.5),method='SLSQP',bounds = bnds, constraints = cnst,tol=1e-10)
        
        # saving results
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        if do_print:
            print(res.message)

            print(f'LM: {opt.LM:.4f}')
            print(f'HM: {opt.HM:.4f}')
            print(f'LF: {opt.LF:.4f}')
            print(f'HF: {opt.HF:.4f}')

        return opt

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        
        # setting up parameters
        par = self.par
        sol = self.sol
    
        # creating vectors for results
        par.lH_vec = np.zeros(len(par.wF_vec))

        #Loop through different values of wF
        for i_w, wF in enumerate(par.wF_vec):
            
            # changing wF
            par.wF = wF

            if discrete: # solving with discrete method
            
                # solve for discrete choice set
                opt = self.solve_discrete()

            else: # solving with continuous method
                
                # solve for continuous choice set
                opt = self.solve_continuous()
            
            # saving results
            par.lH_vec[i_w] = np.log(opt.HF/opt.HM)

            sol.HM_vec[i_w] = opt.HM
            sol.HF_vec[i_w] = opt.HF
            sol.LF_vec[i_w] = opt.LF
            sol.LM_vec[i_w] = opt.LM
            
        


    def run_regression(self):
        """ run regression """
        
        # setting up parameters
        par = self.par
        sol = self.sol
        
        # running regression
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None,do_print=False):
        """ estimate alpha and sigma """
        
        # setting up parameters
        par = self.par
        sol = self.sol

        # setting up objective function
        def obj(x,self):

            par = self.par
            sol = self.sol
            
            #Initial parameters
            par.alpha = x[0]
            par.sigma = x[1]
            
            # solve optimal choice set, account for different wF
            self.solve_wF_vec(discrete=False)
            
            #Run regression for beta_0 and beta_1
            self.run_regression()

            return (par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
        
        # setting bounds for alpha and sigma
        bnds = ((0,1),(0,5))
        
        #Minimize objective function for alpha and sigma
        res = optimize.minimize(obj,x0=(0.5,0.5),method='Nelder-Mead',bounds = bnds,args=(self,))
        
        # saving results of alpha and sigma
        sol.alpha_hat = res.x[0]
        sol.sigma_hat = res.x[1]

        if do_print:
            print(res.message)
            print(f'alpha_hat: {res.x[0]:.4f}')
            print(f'sigma_hat: {res.x[1]:.4f}')

            print(f'beta0_hat: {sol.beta0:.4f}')
            print(f'beta1_hat: {sol.beta1:.4f}')
            print(f'Termination value: {obj(res.x,self):.4f}')

    def estimate_(self,sigma=None,mu=None,do_print=False):
        """ estimate mu and sigma """
        
        # setting up parameters
        par = self.par
        sol = self.sol

        # create objective function
        def obj(x,self):

            par = self.par
            sol = self.sol

            par.sigma = x[0]
            par.mu = x[1]

            self.solve_wF_vec(discrete=False)

            self.run_regression()

            return (par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
        
        # bounds for sigma and mu
        bnds = ((0,5),(0,24))
        
        # optimize fit with mu and sigma
        res = optimize.minimize(obj,x0=(1.25,12),method='Nelder-Mead',bounds = bnds,args=(self,))
                
        # saving results
        sol.mu_hat = res.x[0]
        sol.sigma_hat = res.x[1]

        if do_print:
            print(res.message)
            print(f'sigma_hat: {res.x[0]:.4f}')
            print(f'mu_hat: {res.x[1]:.4f}')

            print(f'beta0_hat: {sol.beta0:.4f}')
            print(f'beta1_hat: {sol.beta1:.4f}')
            print(f'Termination value: {obj(res.x,self):.4f}')
            
