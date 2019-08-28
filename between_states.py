# this contains things for transition between states

import numpy as np


from valueFunctions import valuefunction
from interp_my import interpolate_nostart


class switch:
    
    def __init__(self,options,offset=None):
        self.options = options
        self.v0, self.i0 = self.v0def(options)
        
        assert self.v0 == self[self.i0]
        
        self.outcomes, self.outcomes_i = self.elem_outcomes(self.options)
        
        
        
        
        if offset is None:        
            offset = [Offset()]*len(options)
        offset = [o if isinstance(o,Offset) else Offset() for o in offset]
        self.offset_list = offset
        
        
        o_i = dict(zip(self.outcomes,self.outcomes_i))
        
        self.o_i = o_i
        
        self.offset_prices = dict().fromkeys(self.outcomes,None)
        
        # this block collects indices
        for d in self.offset_prices.keys():
            if type(o_i[d]) is int:
                self.offset_prices[d] = self.offset_list[o_i[d]].price
            else:
                assert type(o_i[d]) is list
                assert  len(o_i[d]) > 1
                self.offset_prices[d] = self.offset_list[o_i[d][0]].price + \
                                        self.options[o_i[d][0]].offset_prices[d]
        
        
        ''' 
        # this in an alternative definition based on inclusion
        
        self.o2_i = dict().fromkeys(self.outcomes,None)
        
        # note: this thing may fail in case of repeated types
        for io in range(len(self.options)):
            if type(self.options[io]) is str:
                self.o2_i[self.options[io]] = io
            else:
                assert isinstance(self.options[io],switch)
                for d_in in self.options[io].o2_i.keys():
                    i_add = self.options[io].o2_i[d_in]
                    l_add = i_add if type(i_add) is list else [i_add]
                    self.o2_i[d_in] = [io] + l_add
        '''
        
        assert isinstance(options,list), 'Options must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in options!'

    #def get_offset(self,key):
        
    
    
    @staticmethod
    def v0def(opts):        
          
        i0 = 0
        
        while not isinstance(opts[0],str):
            opts = opts[0].options
            if type(i0) is not list: i0 = [i0]
            i0 = i0 + [0]
        
        assert type(opts[0]) is str
        
        return opts[0], i0
    
    def elem_outcomes(self,opts):
        outcomes = [self.v0]
        indices  = [self.i0]
        
        for iopt in range(len(opts)):
            opt = opts[iopt]
            # append string outcomes
            if (type(opt) is str) and (not (opt in outcomes)):
                outcomes = outcomes + [opt]
                indices  = indices  + [iopt]
            
            # then unpack switch objects
            if isinstance(opt,switch):
                outcomes_opt, indices_opt = self.elem_outcomes(opt.options)
                
                for iopt_add in range(len(outcomes_opt)):
                    opt_add = outcomes_opt[iopt_add]
                    ind_add = indices_opt[iopt_add]
                    if (type(opt_add) is str) and (not (opt_add in outcomes)):
                        outcomes = outcomes + [opt_add]
                        indices  = indices  + [[iopt] + [ind_add]]
            
        return outcomes, indices
    
    
    def __getitem__(self,key):
        if type(key) is int:
            return self.options[key]
        elif isinstance(key,(tuple,list)): 
            assert len(key)>1, "Please unpack all indices"
            try:
                return self[key[0]][key[1:]]
            except:
                raise Exception('unsupported key')
        else:
            raise Exception('unsupported key')
   
    def __probability_lt__(self,vin):
        # this returns list of tuples
        
        
        probs = self.probability(vin)
        output = list()
        
        
        assert len(probs) == len(self.options)
        
        for i in range(len(probs)):
            opt = self.options[i]
            if type(opt) is str:
                output = output + [(opt,probs[i])]
            else:
                assert isinstance(opt,switch)
                output = output + [(opt.__probability_lt__(vin),probs[i])]
                
                
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
    
    
    def elem_probabilities(self,vin):
        # this collects output of __probability_lt__ that has repeatable items
        # into a nice dictionary of ultimate outcomes and their probabilities
        
        lt = self.__probability_lt__(vin)
        
        out = dict().fromkeys(self.outcomes,None)
        
        for outcome_name in self.outcomes:
            out[outcome_name] = sum( [lt_e[1] for lt_e in lt if (lt_e[0] == outcome_name)] )
            
        assert np.all(  np.abs( sum(out.values()) - 1 ) < 1e-6 ) 
        return out
    
    
    
    
    
    
    def probability(self,Vdict):
        return ev_emu(Vdict,self,lambda x : x,return_p=True,no_offset=False)
    
    
    def apply_offset(self,vlist,put_inf=False):
        if self.offset_list is None:
            return vlist
        else:
            return [v if o is None else o.apply(v,put_inf) for (o,v) in zip(self.offset_list,vlist)]
        
        
        

class choice(switch):
    def __init__(self,options,eps,offset=None):
        switch.__init__(self,options,offset=offset)
        self.eps = eps        
        
        

        assert isinstance(options,list), 'Choices must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in choices!'
    
    
class shock(switch):
    def __init__(self,options,ps,offset=None):
        switch.__init__(self,options,offset=offset)
        self.ps      = ps     
        assert np.abs(sum(ps) - 1)<1e-6
        
        
    
    
    
# this is an offset for transitions
class Offset(object):
    def __init__(self,agrid=None,price=0):
        if agrid is None:
            self.has_offset = False
            self.price = price
            self.agrid = agrid
        else:  
            self.has_offset = True
            self.agrid = agrid
            self.price = price
            self.not_feasible = (agrid<price)
            a_new = np.maximum(agrid-price,agrid[0])
            self.i, self.w, acheck = interpolate_nostart(agrid,a_new,agrid,xd_ordered=True)
            assert np.all(np.abs(a_new - acheck) < 1e-2)
    
    def apply(self,vin,put_inf=False):
        if self.has_offset:    
            #vout = self.w[:,np.newaxis]*vin[self.i,:] + (1-self.w[:,np.newaxis])*vin[self.i+1,:]
            vout = at_iw(vin,self.i,self.w) 
            
            #if self.i[0] == 0 and self.i[1] == 0: assert np.all(vout[0,:] == vout[1,:])
            if put_inf: vout[np.where(self.not_feasible),:] = -np.inf
            if self.price==0: assert np.all(vout == vin)
        
            return vout
        else:
            return vin
        
        
def at_iw(v,i,w,get_diag=False):
    w_mat = w[:,np.newaxis]
    return w_mat*v[i,:] + (1-w_mat)*v[i+1,:]
    
        
'''     
if __name__ == "__main__":
    agrid = M0.V[0]['No children, fertile'].grids[0].points
    oo = Offset(agrid,4)
    vout = oo.apply([M0.V[0]['No children, fertile'].V],put_inf=True)
    print(vout)
'''    
   


# this is input for ev_emu, that can be either whole bunch of value functions
# or value functions and coordinate



def ev_emu(vin,transition,mu,no_offset=False,return_p=False):
    
    '''
    at = lambda v : v
    
    if has_at:
        def at(v):
            assert vin.ndim == 2
            v_i  = v[vin.at_i,:]
            v_ip = v[vin.at_i+1,:]
            v_out = vin.at_w[:,np.newaxis]*v_i + (1-vin.at_w[:,np.newaxis])*v_ip
            return v_out
    '''
    
    def v_mu(v,opt,mu): # this returns pair of v and marginal utility
        return v[opt]['V'], mu(v[opt]['c'])
    
    
    if not no_offset:
        apply_to_tuple = lambda x, o : ( o.apply(x[0], put_inf=True) , o.apply(x[1]) )
    else:
        apply_to_tuple = lambda x, o : x
    
    Vbase, EMUbase = zip(
                            *[
                                  apply_to_tuple( 
                                        v_mu(vin,opt,mu), o 
                                                ) 
                                    if (type(opt) is str) else 
                                  apply_to_tuple(
                                        ev_emu(vin,opt,mu,no_offset), o
                                                ) 
                               for opt, o in
                                  zip(transition.options,transition.offset_list)
                              ]
                        )
    
    
    assert all([not np.any(np.isnan(v)) for v in Vbase])
    assert all([not np.any(np.isnan(v)) for v in EMUbase])
    
    Vstart = vin[transition.v0]
    
    if len(Vbase)==1:
        if not return_p:
            return Vstart['V'], mu(Vstart['c'])
        else:
            return [np.ones_like(Vstart['V'])]
    else:
        if type(transition) is choice:
            EV, p = valuefunction.combine_endo(Vbase,transition.eps,return_p=True)
            if return_p: return p
            EMU   = valuefunction.combine_exo(EMUbase,p[1:])
            return EV, EMU
        elif type(transition) is shock:
            if return_p: return [p*np.ones_like(np.ones_like(Vstart['V'])) for p in transition.ps]           
            EV = valuefunction.combine_exo(Vbase,transition.ps[1:])
            EMU  = valuefunction.combine_exo(EMUbase,transition.ps[1:])
            return EV, EMU
        

        
#def simulate_switch(Vdict,transition,ia,wa):
            
   
    
    