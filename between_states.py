# this contains things for transition between states

import numpy as np

class switch:
    
    @staticmethod
    def v0def(opts):        
          
        while not isinstance(opts[0],str):
            opts = opts[0].options
        
        assert type(opts[0]) is str
        return opts[0]

class choice(switch):
    def __init__(self,options,eps):
        self.options = options
        self.eps = eps        
        self.v0 = self.v0def(options)
        
        assert isinstance(options,list), 'Choices must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in choices!'
    
        
        
class shock(switch):
    def __init__(self,options,ps):
        self.options = options
        self.ps      = ps        
        self.v0 = self.v0def(options)
        
        assert np.abs(sum(ps) - 1)<1e-6
        assert isinstance(options,list), 'Options must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in options!'
        
   
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
            EV, p = Vstart.combine_endo(Vbase,transition.eps,return_p=True)
            EMU   = Vstart.combine_exo(EMUbase,p[1:])
            return EV, EMU
        elif type(transition) is shock:
            EV = Vstart.combine_exo(Vbase,transition.ps[1:])
            EMU  = Vstart.combine_exo(EMUbase,transition.ps[1:])
            return EV, EMU