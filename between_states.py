# this contains things for transition between states

import numpy as np


from valueFunctions import valuefunction


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
    
    
        
        

class choice(switch):
    def __init__(self,options,eps):
        self.options = options
        self.eps = eps        
        self.v0 = self.v0def(options)
        self.outcomes = self.elem_outcomes(self.options)
        
        
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
    def __init__(self,options,ps):
        self.options = options
        self.ps      = ps        
        self.v0 = self.v0def(options)
        self.outcomes = self.elem_outcomes(self.options)
        
        assert np.abs(sum(ps) - 1)<1e-6
        assert isinstance(options,list), 'Options must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in options!'
        
    def probability(self,Vdict):
        return [p*np.ones_like(Vdict[self.v0]['V']) for p in self.ps]
   
def ev_emu(Vdict,transition,mu):
    
    Vbase = [Vdict[opt]['V'] if (type(opt) is str)
             else ev_emu(Vdict,opt,mu)[0]
             for opt in transition.options]
    
    
    EMUbase = [mu(Vdict[opt]['c']) if (type(opt) is str)
                else ev_emu(Vdict,opt,mu)[1]
                for opt in transition.options]
    
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
            
   
    
    