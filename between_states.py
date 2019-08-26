# this contains things for transition between states

import numpy as np


from valueFunctions import valuefunction
from interp_my import interpolate_nostart


class switch:
    
    @staticmethod
    def v0def(opts):        
          
        while not isinstance(opts[0],str):
            opts = opts[0].options
        
        assert type(opts[0]) is str
        return opts[0]
    
    def elem_outcomes(self,opts):
        outcomes = [self.v0]
        
        for opt in opts:
            # append string outcomes
            if (type(opt) is str) and (not (opt in outcomes)):
                outcomes = outcomes + [opt]
            
            # then unpack switch objects
            if isinstance(opt,switch):
                outcomes_opt = self.elem_outcomes(opt.options)
                for opt_add in outcomes_opt:
                    if (type(opt_add) is str) and (not (opt_add in outcomes)):
                        outcomes = outcomes + [opt_add]
            
        return outcomes
    
    
    
    def __probability_lt__(self,Vdict):
        # this returns list of tuples
        
        
        probs = self.probability(Vdict)
        output = list()
        
        
        assert len(probs) == len(self.options)
        
        for i in range(len(probs)):
            opt = self.options[i]
            if type(opt) is str:
                output = output + [(opt,probs[i])]
            else:
                assert isinstance(opt,switch)
                output = output + [(opt.__probability_lt__(Vdict),probs[i])]
                
                
        # this unpacks all multi-level tuples
        def get_raw(lt,multiplier=1):
            out = list()
            
            for lt_elem in lt:
                if type(lt_elem[0]) is str:
                    out = out + [(lt_elem[0],multiplier*lt_elem[1])]
                else:
                    out = out + get_raw(lt_elem[0],multiplier=lt_elem[1])
            return out
        
        return get_raw(output)
    
    
    def elem_probabilities(self,Vdict):
        # this collects output of __probability_lt__ that has repeatable items
        # into a nice dictionary of ultimate outcomes and their probabilities
        
        lt = self.__probability_lt__(Vdict)
        
        out = dict().fromkeys(self.outcomes,None)
        
        for outcome_name in self.outcomes:
            out[outcome_name] = sum( [lt_e[1] for lt_e in lt if (lt_e[0] == outcome_name)] )
            
        assert np.all(  np.abs( sum(out.values()) - 1 ) < 1e-6 ) 
        return out
    
    def apply_offset(self,vlist,put_inf=False):
        if self.offset_list is None:
            return vlist
        else:
            return [v if o is None else o.apply(v,put_inf) for (o,v) in zip(self.offset_list,vlist)]
        
        
        

class choice(switch):
    def __init__(self,options,eps,offset=None):
        self.options = options
        self.eps = eps        
        self.v0 = self.v0def(options)
        self.outcomes = self.elem_outcomes(self.options)
        self.offset_list = offset
        

        assert isinstance(options,list), 'Choices must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in choices!'
    
    def probability(self,Vdict):
        
        #Vstart = Vdict[self.v0]        
        Vbase = [Vdict[opt]['V'] if (type(opt) is str)
                 else ev_emu(Vdict,opt,lambda x : x)[0]
                 for opt in self.options]   
        
        if len(self.options) == 1:
            return [np.ones_like(Vdict[self.v0]['V'])]
        else:
            return valuefunction.combine_endo(Vbase,self.eps,return_p=True)[1]
        
    
        
class shock(switch):
    def __init__(self,options,ps,offset=None):
        self.options = options
        self.ps      = ps        
        self.v0 = self.v0def(options)
        self.outcomes = self.elem_outcomes(self.options)
        self.offset_list = offset
        
        assert np.abs(sum(ps) - 1)<1e-6
        assert isinstance(options,list), 'Options must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in options!'
        
    def probability(self,Vdict):
        return [p*np.ones_like(Vdict[self.v0]['V']) for p in self.ps]
    
    
    
    
class Offset(object):
    def __init__(self,agrid,price):
        self.price = price
        self.not_feasible = (agrid<price)
        a_new = np.maximum(agrid-price,agrid[0])
        self.i, self.w, acheck = interpolate_nostart(agrid,a_new,agrid,xd_ordered=True)
        assert np.all(np.abs(a_new - acheck) < 1e-2)
    
    def apply(self,vin,put_inf=False):
              
        vout = self.w[:,np.newaxis]*vin[self.i,:] + (1-self.w[:,np.newaxis])*vin[self.i+1,:]
            
            
        #if self.i[0] == 0 and self.i[1] == 0: assert np.all(vout[0,:] == vout[1,:])
        if put_inf: vout[np.where(self.not_feasible),:] = -np.inf
        if self.price==0: assert np.all(vout == vin)
        
        
        return vout
        
        
'''     
if __name__ == "__main__":
    agrid = M0.V[0]['No children, fertile'].grids[0].points
    oo = Offset(agrid,4)
    vout = oo.apply([M0.V[0]['No children, fertile'].V],put_inf=True)
    print(vout)
'''    
   
def ev_emu(Vdict,transition,mu):
    
    Vbase = [Vdict[opt]['V'] if (type(opt) is str)
             else ev_emu(Vdict,opt,mu)[0]
             for opt in transition.options]
    
    
    EMUbase = [mu(Vdict[opt]['c']) if (type(opt) is str)
                else ev_emu(Vdict,opt,mu)[1]
                for opt in transition.options]
    
    
    
    Vbase = transition.apply_offset(Vbase,put_inf=True)
    EMUbase = transition.apply_offset(EMUbase,put_inf=False)
        
    assert all([not np.any(np.isnan(v)) for v in Vbase])
    assert all([not np.any(np.isnan(v)) for v in EMUbase])
    
    Vstart = Vdict[transition.v0]
    
    if len(Vbase)==1:
        return Vstart['V'], mu(Vstart['c'])
    else:
        if type(transition) is choice:
            EV, p = valuefunction.combine_endo(Vbase,transition.eps,return_p=True)
            EMU   = valuefunction.combine_exo(EMUbase,p[1:])
            return EV, EMU
        elif type(transition) is shock:
            EV = valuefunction.combine_exo(Vbase,transition.ps[1:])
            EMU  = valuefunction.combine_exo(EMUbase,transition.ps[1:])
            return EV, EMU
        

        
#def simulate_switch(Vdict,transition,ia,wa):
            
   
    
    