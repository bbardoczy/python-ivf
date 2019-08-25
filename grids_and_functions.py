# this contains choices for grids and functional forms
import numpy as np
from numba import njit

class setupClass:
    def __init__(self,pars):
        self.pars = pars
        self.define_agrid()
        self.define_u()
        self.define_trends()
        self.define_grids_and_transitions()
        self.define_labor_income()
        self.define_transitions()
        
        
        
    def define_agrid(self):
        from generate_grid import generate_agrid
        self.agrid = generate_agrid(n = self.pars['na'], amax=self.pars['amax'])  

    
    def define_u(self):
        import uenv
        
        sigma, u_kid_c, u_kid_add = self.pars['sigma'], self.pars['u_kid_c'][0], self.pars['u_kid_add'][0]
        u_kid_c_2, u_kid_add_2 = self.pars['u_kid_c'][1], self.pars['u_kid_add'][1]
        
        
        uselog = True if sigma == 1 else False
        
        if uselog:        
            factor = 1
            factor_2 = 1
            @njit
            def u_nok(x):
                return np.log(x)        
            @njit
            def u_k(x):
                return np.log(x) + u_kid_c + u_kid_add
            @njit
            def u_kk(x):
                return np.log(x) + u_kid_c_2 + u_kid_add_2
            
        else:
            factor = np.exp(u_kid_c*(1-sigma))  
            factor_2 = np.exp(u_kid_c_2*(1-sigma))   
            @njit
            def u_nok(x: np.float64) -> np.float64:
                return (x**(1-sigma))/(1-sigma)        
            @njit
            def u_k(x: np.float64) -> np.float64:
                return factor*(x**(1-sigma))/(1-sigma) + u_kid_add
            @njit
            def u_kk(x: np.float64) -> np.float64:
                return factor_2*(x**(1-sigma))/(1-sigma) + u_kid_add_2
            
        
        self.u =  {
                'Single':    u_nok,
                'No children, fertile':    u_nok,
                'One child, out, fertile': u_k,
                'One child, in, fertile':  u_k,
                'Two children, out, fertile': u_kk,
                'Two children, in, fertile':  u_kk
             }
        
        self.mu = {
                'Single': lambda x  : (x**(-sigma)),
                'No children, fertile':    lambda x  : (x**(-sigma)),
                'One child, out, fertile': lambda x  : factor*(x**(-sigma)),
                'One child, in, fertile':  lambda x  : factor*(x**(-sigma)),
                'Two children, out, fertile': lambda x  : factor_2*(x**(-sigma)),
                'Two children, in, fertile':  lambda x  : factor_2*(x**(-sigma))
             }
        
        self.mu_inv = {
                  'Single': lambda MU : (MU**(-1/sigma)),
                  'No children, fertile':    lambda MU : (MU**(-1/sigma)),  
                  'One child, out, fertile': lambda MU : ((MU/factor)**(-1/sigma)),
                  'One child, in, fertile':  lambda MU : ((MU/factor)**(-1/sigma)),
                  'Two children, out, fertile':  lambda MU : ((MU/factor_2)**(-1/sigma)),
                  'Two children, in, fertile':  lambda MU : ((MU/factor_2)**(-1/sigma)),
                 }
            
        
        self.ue = {
                'Single':   uenv.create(self.u['Single'],False),
                'No children, fertile':    uenv.create(self.u['No children, fertile'],False),
                'One child, out, fertile': uenv.create(self.u['One child, out, fertile'],False),
                'One child, in, fertile':  uenv.create(self.u['One child, in, fertile'],False),
                'Two children, out, fertile': uenv.create(self.u['Two children, out, fertile'],False),
                'Two children, in, fertile':  uenv.create(self.u['Two children, in, fertile'],False)
             }
        

    @staticmethod   
    def define_polynomial_trend(coefs,ts,fun=lambda x: x):
        T = np.ones_like(ts)
        trend = np.zeros_like(ts)
        
        for a in coefs:
            trend += a*T
            T     *= ts
            
        return fun(trend)
    
    def define_trends(self,exp_g=True,normalize_T=True):
        
        pars = self.pars
        
        zcoefs = [pars['a_z0'],pars['a_z1'],pars['a_z2']]
        gcoefs = [pars['a_g0'],pars['a_g1']]
        
        trange = np.arange(0,pars['T'])
        
        if normalize_T:
            trange = trange/np.mean(trange)
        self.ztrend = self.define_polynomial_trend(zcoefs,trange)
        
        self.gtrend = self.define_polynomial_trend(gcoefs,trange)
        
        if exp_g:
            self.gtrend = np.exp(self.gtrend)
            
    
    
    
    def define_grids_and_transitions(self):
        # this takes whole pars as I did not figure out a better way
        
        
        from generate_grid import generate_zgs
        
        pars = self.pars
        
        a = dict(sigma_z_init=pars['sigma_z_init'],sigma_z=pars['sigma_z'],nz=pars['nz'],
                     sigma_g=pars['sigma_g'],rho_g=pars['rho_g'],ng=pars['ng'],smin=0,smax=pars['smax'],ns=pars['ns'],
                     T=pars['T'],mult=self.gtrend)
        
        zgs_GridList_nok,    zgs_MatList_nok   = generate_zgs(**a)
        zgs_GridList_k_in,  zgs_MatList_k_in   = generate_zgs(**a,fun = lambda g : np.maximum( g - pars['g_kid'][0], 0 ) )
        zgs_GridList_k_out, zgs_MatList_k_out  = generate_zgs(**a,fun = lambda g : -pars['delta_out'] )
        zgs_GridList_kk_in,  zgs_MatList_kk_in   = generate_zgs(**a,fun = lambda g : np.maximum( g - pars['g_kid'][1], 0 ) )
        zgs_GridList_kk_out, zgs_MatList_kk_out  = generate_zgs(**a,fun = lambda g : -pars['delta_out'] )
        
        
        
        self.zgs_Grids =    {
                            'Single': zgs_GridList_nok,
                            'No children, fertile': zgs_GridList_nok,
                            'One child, out, fertile': zgs_GridList_k_out,
                            'One child, in, fertile': zgs_GridList_k_in,
                            'Two children, out, fertile': zgs_GridList_kk_out,
                            'Two children, in, fertile': zgs_GridList_kk_in
                            }
        
        self.zgs_Mats  =    {
                            'Single': zgs_MatList_nok,
                            'No children, fertile': zgs_MatList_nok,
                            'One child, out, fertile': zgs_MatList_k_out,
                            'One child, in, fertile': zgs_MatList_k_in,
                            'Two children, out, fertile': zgs_MatList_kk_out,
                            'Two children, in, fertile': zgs_MatList_kk_in
                            }
        

    
    def define_labor_income(self):
        
        pars = self.pars
        
        
        
        
        
        self.labor_income = {
                        'Single': lambda grid, t : np.exp(grid[:,0] + grid[:,2] + self.ztrend[t]).T,
                        'No children, fertile':    lambda grid, t : np.exp(grid[:,0] + grid[:,2] + self.ztrend[t]).T,
                        'One child, out, fertile': lambda grid, t : pars['phi_out']*np.exp(grid[:,0] + grid[:,2] - pars['z_kid'][0]  + self.ztrend[t]).T,
                        'One child, in, fertile':  lambda grid, t : pars['phi_in']*np.exp(grid[:,0] + grid[:,2]  - pars['z_kid'][0]  + self.ztrend[t]).T,
                        'Two children, out, fertile': lambda grid, t : pars['phi_out']*np.exp(grid[:,0] + grid[:,2] - pars['z_kid'][1]  + self.ztrend[t]).T,
                        'Two children, in, fertile':  lambda grid, t : pars['phi_in']*np.exp(grid[:,0] + grid[:,2]  - pars['z_kid'][1]  + self.ztrend[t]).T,
                            }
        

    
    def define_transitions(self,new_from_out=False):
        pars = self.pars


        from between_states import shock, choice, Offset
        
        offset = [None,Offset(self.agrid,1e-3)]
            
        
        new_baby = shock(['One child, out, fertile',"One child, in, fertile"],[1-pars['pback'],pars['pback']])
        new_baby_2 = shock(['Two children, out, fertile',"Two children, in, fertile"],[1-pars['pback'],pars['pback']])



        if new_from_out:
            oc_out =  choice([shock(["One child, out, fertile","One child, in, fertile"],[1-pars['pback'],pars['pback']]),'Two children, out'],pars['eps'],offset)
        else:
            oc_out =  shock(["One child, out, fertile","One child, in, fertile"],[1-pars['pback'],pars['pback']])
        
        
        self.transitions_t = list()
        
        for t in range(self.pars['T']):
            
            
            
            transitions = {             'Single'      :    shock(['Single', 'No children, fertile'],[1-pars['pmar'],pars['pmar']]),
                                'No children, fertile':    choice(["No children, fertile",new_baby],pars['eps'],offset),
                             'One child, out, fertile':    oc_out,
                              'One child, in, fertile':    choice(["One child, in, fertile",new_baby_2],pars['eps'],offset),
                          'Two children, out, fertile':    shock(["Two children, out, fertile","Two children, in, fertile"],[1-pars['pback'],pars['pback']]),
                           'Two children, in, fertile':    choice(["Two children, in, fertile"],pars['eps'])}
            self.transitions_t.append(transitions)
    