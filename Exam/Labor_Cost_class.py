import numpy as np
from scipy.optimize import minimize
from types import SimpleNamespace


class Labor_Cost_class:

    def __init__(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        par.eta = 0.5
        par.w = 1.0
        par.kappa = 1.0
        par.rho = 0.9
        par.sigma_e = 0.1
        par.T = 120
        par.K = 10_000
        par.R = (1.01)**(1/12)
        par.iota = 0.01
        par.delta = 0.075
        par.seed = False
        par.ell_rule = 'standard'

        sol.ell = np.nan
        sol.ell_ana = np.nan
        sol.delta = np.nan

        sim.draws = np.zeros((par.K,par.T))
        sim.kappa_path = np.zeros((par.K,par.T))
        sim.ell = np.zeros((par.K,par.T))
        sim.discounted_profit = np.zeros((par.K))
        sim.ell_star = np.zeros((par.K,par.T))
        sim.H = np.nan
    
    def analytical_solution(self):

        par = self.par
        sol = self.sol

        sol.ell_ana =  ((1-par.eta)*par.kappa/par.w)**(1/par.eta)

    def analytical_solution_path(self,k):

        par = self.par
        sim = self.sim

        sim.ell_star[k,:] = ((1-par.eta)*sim.kappa_path[k]/par.w)**(1/par.eta)
    
    def calc_profit(self,ell):

        par = self.par

        return par.kappa*ell**(1-par.eta) - par.w*ell
    
    def solve_numerically(self):

        par = self.par
        sol = self.sol

        def obj(ell):
            
            return -self.calc_profit(ell)
        
        res = minimize(obj,0.1,method='BFGS')

        sol.ell = res.x.item()

    def compare_num_ana(self):

        par = self.par
        sol = self.sol

        self.analytical_solution()
        self.solve_numerically()

        print(f'Numerical solution with {par.kappa=}: {sol.ell:.6f}')
        print(f'Analytical solution with {par.kappa=}: {sol.ell_ana:.6f}')

    def draws(self):

        par = self.par
        sim = self.sim

        if par.seed:
            np.random.seed(1997)
        sim.draws = np.random.normal(-0.5*par.sigma_e**2,par.sigma_e,size=(par.K,par.T))

    def create_kappa_path(self):

        par = self.par
        sol = self.sol
        sim = self.sim

        self.draws()

        for k in range(par.K):
            for t in range(par.T):
                if t == 0:
                    sim.kappa_path[k,t] = np.exp(par.rho*np.log(1) + sim.draws[k,t])
                else:
                    sim.kappa_path[k,t] = np.exp(par.rho*np.log(sim.kappa_path[k,t-1]) +  + sim.draws[k,t])

    def h(self,k):

        par = self.par
        sol = self.sol
        sim = self.sim

        self.ell_rule(k)

        ell = sim.ell[k]
        ell_lag = np.roll(ell,1)
        ell_lag[0] = 0.0

        profits = sim.kappa_path[k]*ell**(1-par.eta)-par.w*ell-par.iota*(ell!=ell_lag)

        sim.discounted_profit[k] = np.sum(profits*par.R**(-np.arange(par.T)))

    def ell_rule(self,k):

        par = self.par
        sol = self.sol
        sim = self.sim

        self.analytical_solution_path(k)

        for t in range(par.T):
            if t==0:
                ell_lag = 0.0
            else:
                ell_lag = sim.ell[k,t-1]
            
            if par.ell_rule == 'standard':
                diff = abs(ell_lag-sim.ell_star[k,t])

                if diff > par.delta:
                    sim.ell[k,t] = sim.ell_star[k,t]
                else:
                    sim.ell[k,t] = ell_lag

            elif par.ell_rule == 'new':

                # profit if ell is changed
                profit_star = sim.kappa_path[k,t]*sim.ell_star[k,t]**(1-par.eta) - par.w*sim.ell_star[k,t] - par.iota
                # profit if ell is unchanged
                profit_lag = sim.kappa_path[k,t]*ell_lag**(1-par.eta) - par.w*ell_lag

                if profit_star >= profit_lag:
                    sim.ell[k,t] = sim.ell_star[k,t]
                else:
                    sim.ell[k,t] = ell_lag

    def H(self):

        par = self.par
        sol = self.sol
        sim = self.sim

        self.create_kappa_path()
        k = 0
        out = 0

        for k in range(par.K):
            self.h(k)
        
        sim.H = np.sum(sim.discounted_profit)/par.K