# this is a general class for value functions
import numpy as np
#

# combination of grids and values



# these are classes for grids

class grid_dim():
    def __init__(self,points,interpolant_name):
        self.points = points
        self.interpolant_name = interpolant_name
        
    def __getitem__(self,key): return self.points[key]
    def shape(self): return self.points.shape[0] # this is important
    def __mul__(self,other): return self.points*other
    def interp(v,xq): raise Exception('Not implemented!')



class grid_dim_linear(grid_dim):
    def __init__(self,points):
        grid_dim.__init__(self,points,interpolant_name='linear')
        
    def interp(self,v,xq):
        # this should eat vector / matrix v and vector / array xq
        from interp_my import interpolate_nostart as intp
        
        assert v.ndim  == 1, "multiple v are not supported yet"
        assert xq.ndim == self.points.ndim, 'shape problems?'
        
        return intp(self.points,xq,v)
        
class grid_dim_discrete(grid_dim):
    def __init__(self,points):
        grid_dim.__init__(self,points,interpolant_name='none')
        
    def interp(self,v,xq):
        raise Exception('This is not supposed to be interpolated!')
        


# these are classes for value functions
        
        
# generic function with values given by grids
class gridval():
    def __init__(self,grids,values,check=True):
        
        assert isinstance(grids,  list), "First argument should be list ([])!"
        assert isinstance(values, dict), "Second argument should be dict ({'a':b})!"
        
        self.grids = grids
        self.values = values
        
        shp_list = list()
     
        
        for k in self.grids:
            shp_list = shp_list + [k.shape()] # the last dimension is used as length
            
        self.shp = tuple(shp_list)
        
        if check:
            self.check_integrity()
            
    def __getitem__(self, key): return self.values[key]
        
    def check_integrity(self):    
        for j in self.grids:
            assert isinstance(j,grid_dim)
        
        for k in self.values.keys():
            assert np.shape(self.values[k]) == self.shp, "True shapes are {} and {}".format(np.shape(self.values[k]), self.shp)
            
            
# basic value function (collection of V and decision rules)
class valuefunction(gridval):
    
    def __init__(self,grids,values,time,description="",check=True):
        gridval.__init__(self,grids,values,check=True)
        assert ('V' in values), 'values should have V'
        self.V = self.values['V']
        self.time = time
        self.description = description
    
    
    def __getitem__(self, key):
        if isinstance(key,str):
            return gridval.__getitem__(self, key)
        
        elif isinstance(key,list):
            
            assert isinstance(key[0],(str,list))
            assert isinstance(key[1],(str,list))
            assert callable(key[2])
            
            v0 = self[key[0]]
            v1 = self[key[1]]
            
            return key[2](v0,v1)
        
        else: # treat this as if we ask about V
            return self.values['V'][key]
    
            
    
    def integrate(self,integrator,*args):
        return integrator(self,*args)
    
    
    
    
    def plot_value(self,field, iz = None):
        # this plots value and adds time and description
        import matplotlib.pyplot as plt
        
        # this is not super general ofc
        if iz is None:
            iz = int(self.grids[1].shape()/2)
        
        label = ("T = " + str(self.time) + ", iz = " + str(iz))
        
        if self.description is not None:
            label = label + ", " + self.description
            
        y = self[field][:,iz]
        x = self.grids[0].points
        
        
        plt.plot(x, y, label=label)
        
        #return y    
    
    
    
    
    def plot_diff(self,other,field,iz=None,between="-",operation=np.subtract):
        import matplotlib.pyplot as plt
        
        if iz is None:
            iz = int(self.grids[1].shape()/2)
            
        label1 = ("T" + str(self.time) + "iz" + str(iz))
        if self.description is not None:
            label1 = label1 + "(" + self.description + ")"
            
        label2 = ("T" + str(other.time) + "iz" + str(iz))
        if self.description is not None:
            label2 = label2 + "(" + other.description + ")"
            
            
        label = label2 + between + label1
        
        y_other = other[field][:,iz]
        y_self = self[field][:,iz]
        
        y = operation(y_other,y_self)
        y0 = np.zeros_like(y)
        x = self.grids[0].points
        
        
        plt.plot(x, y, label=label)
        plt.plot(x, y0,label="zeros")
        
        #return y
    
    
    def combine(self,vlist=None,ps=None,eps=None,field='V', fun = lambda x : x, return_p = False):
        # combines multiple value functions
        # if plist is supplied 
        # ps contains probabilities or eps contains taste shocks standard deviations
        
        assert (ps is None) or (eps is None)
        
        if vlist is None:
            return self[field]
        
        
        if not isinstance(vlist,list):
            vlist = [vlist]
            
        v_in = [fun(self[field])] + [ fun(i[field]) if type(i) is valuefunction else fun(i) for i in vlist ]
            
            
        if ps is not None:
            # exogenous combination
            if (not isinstance(ps,list)):
                ps = [ps]
            return self.combine_exo(v_in,ps)
        else:
            return self.combine_endo(v_in,eps,return_p = return_p)
        
    
    def combineV(self,*args,**kwargs):
        val = self.combine(*args,**kwargs)
        return valuefunction(self.grids,{'V':val},self.time,description='EV')
            
    
            
       
    @staticmethod    
    def combine_endo(vlist,eps,return_p=False):
        vmax = np.maximum(*vlist)
        
        
        p = [None]*len(vlist)
        
        if eps > 1e-6:
            S = 0.0
            
            for i in range(len(vlist)):
                p[i] = np.exp((vlist[i] - vmax)/eps)
                S += p[i]
            
            p = [j/S for j in p]
            print('yes')
            assert np.all(S >= 1.0)
            
        else:
            S = 1.0 # does not matter
            
            
            for i in range(len(vlist)):
                p[i] = np.float64(vlist[i]==vmax)
                
                
        # test
        if len(vlist)==2:
            v0 = vmax + eps*np.log(S) - eps*np.log(len(vlist)) 
            from vf_tools import smooth_max
            assert np.all(np.abs(v0 - smooth_max(vlist[0],vlist[1],eps))<1e-6)
                
        if not return_p:
            return vmax + eps*np.log(S) - eps*np.log(len(vlist))
        else:
            return vmax + eps*np.log(S) - eps*np.log(len(vlist)), p
    
    
    
    
    @staticmethod
    def combine_exo(vlist,plist_wo_0):
        
        if type(plist_wo_0[0]) is np.ndarray:
            Psum = np.zeros_like(plist_wo_0[0])
            
            for i in range(len(plist_wo_0)):
                Psum += plist_wo_0[i]
        else:
            Psum = np.sum(plist_wo_0)
                
        
        p0 = 1-Psum
        
        assert np.all(p0 >= 0) and np.all(p0 <= 1)
        plist = [p0] + plist_wo_0
        
        assert len(vlist) == len(plist)
        
        S = 0.0
        for i in range(len(vlist)):
            S += plist[i]*vlist[i]
        
        return S
        
        
        
            
            

