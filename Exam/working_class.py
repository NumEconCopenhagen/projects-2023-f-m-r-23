import numpy as np
from scipy.optimize import minimize
from types import SimpleNamespace
from scipy.optimize import minimize, minimize_scalar, brentq


class working_class():
    def __init__(self,do_print=True):
        """ create the model """
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        
        self.setup()

    def setup(self):
        """ baseline parameters """

        par = self.par
        sol = self.sol
        
        #Set up parameters
        par.alpha=0.5
        par.kappa=1.0
        par.nu = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.3
        par.G = 1.0   
        
        par.with_G = False
        
        par.sigma = 1.001
        par.rho=1.0001
        par.epsilon = 1.0
            
        par.simT = 50 # length of simulation
    
    def utility(self,L):
        """ First utility function  """

        par = self.par
        
        if par.with_G == False:
            C = par.kappa+(1-par.tau)*par.w*L
            utility = np.log(C**par.alpha*par.G**(1-par.alpha))-par.nu*(L**2/2)
        
        elif par.with_G == True:
            C = par.kappa+(1-par.tau)*par.w*L
            G = par.tau*par.w*L*((1-par.tau)*par.w)
            utility = np.log(C**par.alpha*G**(1-par.alpha))-par.nu*(L**2/2)
        
          
        return utility
        
    def solve(self):
        """ Solve the non-general case  """
        
        par = self.par
        
        opt = SimpleNamespace()
        
        # creating objective function for optimize
        def obj(x):
            return -self.utility(x[0])
        
        #Optimize
        res = minimize(obj,x0 = (12),method='BFGS',tol=1e-10)
        
        #Saving results
        opt.L = res.x[0]
        
        return opt
    
    def analytical_sol(self,w):
        """ Analytical solution  """
    
        par = self.par
        
        w_tau = (1-par.tau)*w
        
        L = (-par.kappa+np.sqrt(par.kappa**2+4*(par.alpha/par.nu)*w_tau**2))/(2*w_tau)
        
        return L

    def utility_general(self,L):
        """ General utility function  """
    
        par = self.par
        
        C = par.kappa+(1-par.tau)*par.w*L
        G = par.G
        
        first = par.alpha*C**((par.sigma-1)/par.sigma)
        second = (1-par.alpha)*G**((par.sigma-1)/(par.sigma))
        
        comb = (((first+second)**(par.sigma/(par.sigma-1)))**(1-par.rho)-1)/(1-par.rho)
        last = par.nu*(L**(1+par.epsilon)/(1+par.epsilon))
                                         
        utility = comb - last
        
        return utility
    
    def solve_general(self):
        """ Solve the utility function for the general case  """
        
        par = self.par
        sol = self.sol
        
        opt = SimpleNamespace()
        
        # creating objective function for optimize
        def obj(x):
            return -self.utility_general(x)
        
        #Optimize
        res = minimize_scalar(obj, bounds=(0, 24), method='bounded',tol=1e-10)
        
        #Saving results
        opt.L = res.x
        sol.L = res.x
        
        return opt
    
    
    def solve_q5(self):
        """ Solve Question 5  """
        
        def obj(G):
            par.G = G
            model.solve_general()
        
            return par.G-par.tau*par.w*sol.L
    
        res = brentq(obj,-100,100)
    
        return res
    