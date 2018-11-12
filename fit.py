# -*- coding: utf-8 -*-
"""
Created on 11/8/18 Updated:
Written By: Jonathan D B Van Schenck

This module is used to fit an absorption spectrum with a vibronic
progressions of the form at the end of "derivation.pdf"--Either using a 
single VP, or the sum of two VPs. This is most useful for fitting solution 
absorption, where JnnpEv is fixed at zero, but is usable for true crystalline 
systems. 


Dependancies:
lineshape:         Used for "absLineMu" which is lineshape for fitting
numpy:             Used for array handling
matplotlib:        Uses pyplot for basic plotting
scipy.optimize:    Uses least_squares for curve fitting


Classes
singleFit:         Fits a single absorption spectrum with a single vibronic
                    progression
doubleFit:         Fits a single absorption spectrum with two vibronic
                    progressions
"""
import lineshape as ls
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

class singleFit:
    """
    This module fits absorption data with a single vibronic progression of the 
    form as the last equation of "derivation.pdf"
    
    Inputs Required
    ev:         1d Numpy array holding the photon energies of absorption (in eV)
    spec:       1d Numpy array holding the absorption at each photon energy
    fun:        A function "fun(x,mu,sig)" for 'lineshape.absLineMu'. x is 
                 a numpy array holding the domain, mu is the center of the
                 distribution, and sig is the broadening (usually FWHM).
    
    Attributes:
    ev:         1d Numpy array holding the photon energies of absorption (in eV)
    n:          Integer holding the number of absorption data points
    spec:       1d Numpy array holding the absorption at each photon energy
    spec2:      1d Numpy array holding the absorption divided by photon energy
                 (used in fitting proceedure)
    fun:        A function "fun(x,mu,sig)" for 'lineshape.absLineMu'. x is 
                 a numpy array holding the domain, mu is the center of the
                 distribution, and sig is the broadening (usually FWHM).       
    paramNames: 1d Numpy array holding the names of each parameter:
                    a:      The area of the entire lineshape
                    Ex:     The energy of the uncoupled exciton in eV
                    Ev:     The quantum of vibrational energy for the molecule in eV
                    S:      The HR factor of the molecule
                    sig0:   The broadening of the 0-0 absorption peak in eV 
                             (Usually full width)
                    Dsig:   The progressive broadening % accrued for higher 
                             vibrational states
                    JnnpEv: The ratio of the nearest neighbor coupling to the 
                             vibrational quantum
    iparam:     1d numpy array holding the initial guess for paramter values
                 (must be specified using .initializeFitParams BEFORE the fit
                  can be performed). Structure:
                  [a, Ex, Ev, S, sig0, Dsig, JnnpEv]
    param:      1d numpy array holding the resulting parameter values after 
                 fitting. Structure:
                  [a, Ex, Ev, S, sig0, Dsig, JnnpEv]
    which:      1d bool array holding specifying which of the parameters will
                 be allowed to varrying during fitting. Default is to allow all
                 parameters to varry. Can be modified using .freezeFitParams.
                 Structure:
                  [a?,Ex?,Ev?,S?,sig0?,Dsig?,JnnpEv?]
    bound:      2d numpy array holding the paramater bounds to be used during 
                 fitting. If parameters have been frozen by using .freezeFitParam
                 method, then bound will only contain bounds for the parameters
                 which are used during fitting. i.e. bound.shape[1]=nf. 
                 Bound[0] is the lower bound and bound[1] is the upper
                 bound. Note, iparam[i] MUST be in range (bound[0][i],bound[1][i])
                 Structure when no parameters are frozen:
                  [[a_,Ex_,Ev_,S_,sig0_,Dsig_,JnnpEv_],
                   [a^,Ex^,Ev^,S^,sig0^,Dsig^,JnnpEv^]]
    nf:         Value holding the number of parameters allowed to varry
    fitMask:    1d bool array specifying which reflectance data points to be 
                 used during fitting. Default is to use all reflectance data.
                 Can be modified using .createFitRegion
    plotMask:   1d bool array specifying which reflectance data points to be 
                 used during plotting. Default is to use all reflectance data.
                 Can be modified using .createPlotRegion
                 
    
    Best Practice for Use:
          1)  Call singleFit class and provide ev, spec, and fun
         2a)  Specify fit region (.createFitRegion)
          3)  Specify plot region (.createPlotRegion)
     Opt  4)  Freeze parameters NOT used during fitting (.freezeFitParams)
          5)  Provide inital guess for parameter values (.initializeFitParams)
          6)  Set bounds on free fit parameters (.createFitParamBounds)
          7)  Perform Fit (.performFit)
     opt/ 8)  Plot resuts (.plot)
     opt\ 9)  Print fit results (.printParam)
        
    
    """
    def __init__(self,ev,spec,fun = ls.Glorz):
        """
        This method sets up the fitting proceedure
        Input
        ev:         1d Numpy array holding the photon energies of absorption (in eV)
        spec:       1d Numpy array holding the absorption at each photon energy
        fun:        A function "fun(x,mu,sig)" for 'lineshape.absLineMu'. x is 
                     a numpy array holding the domain, mu is the center of the
                     distribution, and sig is the broadening (usually FWHM).
        """
        self.ev = ev
        self.n = ev.shape[0]
        self.spec = spec
        self.spec2 = spec/ev
        self.which = np.full(7,True,dtype='bool')
        self.iparam = np.zeros(7)
        self.param = np.copy(self.iparam)
        self.paramNames = np.array(['a','Ex','Ev','S','sig0','Dsig','JnnpEv'])
        self.plotMask = np.full(self.n,True,dtype='bool')
        self.fitMask = np.full(self.n,True,dtype='bool')
        self.fit = 0
        self.bound = 0
        self.nf = 7
        self.fun = fun
        
    def createFitRegion(self,mini,maxi,eV=True):
        """
        Function creates a boolean mask for data to be use during fitting to
        select which data points to fit by. Allows one to select a range of
        either photon energies, or wavelengths inside which to fit.
        
        Input
        mini:       Float: left bound on fit range
        maxi:       Float: right bound on fit range
        ev:         Boolean: when true, fit range is over photon energy (in eV)
                     when false, fit range is selected over wavelength. Note
                     that fits are always assume energy domain, so wavelengths 
                     will later be converted into energies.
        """
        if eV:
            self.fitMask = ls.vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.fitMask = ls.vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
            
    def createPlotRegion(self,mini,maxi,eV=True):
        """
        Function creates a boolean mask for data to be use during plotting to
        select which data points to plot by. Allows one to select a range of
        either photon energies, or wavelengths inside which to fit.
        
        Input
        mini:       Float: left bound on fit range
        maxi:       Float: right bound on fit range
        ev:         Boolean: when true, plot range is over photon energy (in eV)
                     when false, plot range is selected over wavelength. Note
                     that plots are always assume energy domain, so wavelengths
                     will later be converted into energies.
        """
        if eV:
            self.plotMask = ls.vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.plotMask = ls.vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])

    def freezeFitParams(self,a=True,Ex=True,Ev=True,S=True,sig0=True,Dsig=True,JnnpEv=True):
        """
        Function allows user to freeze particular parameters during fitting. 
        By specifying boolean "False" for a parameter, it is not allowed to vary
        during fitting. Structure: [a?,Ex?,Ev?,S?,sig0?,Dsig?,JnnpEv?]
        
        Input
        which:      1d Boolean list/array. Array MUST be the same length as 
                     iparam/param/paramNames. If which[i]==True, than param[i]
                     is allowed to vary during fitting. If which[i]==False, 
                     than param[i] is frozen during fitting.
        """
        self.which = np.array([a,Ex,Ev,S,sig0,Dsig,JnnpEv],dtype='bool')
        self.nf = np.sum(np.ones(7)[self.which])    
    
    def initializeFitParams(self,iparam=-1,a=-1,Ex=-1,Ev=-1,S=-1,sig0=-1,Dsig=-1,JnnpEv=-1):
        """
        Function sets the initial guess for parameter values. Parameter values
        can either be individually specified (default), OR iparam can be 
        directly specified. Either give ALL parameters individually OR iparam.
        Example:
            1) self.initializeFitParams(a=0.1,Ex=2.25,Ev=.17,S=0.7,sig0=0.1,Dsig=0.1,JnnpEv=0.05)
            2) self.initializeFitParams([0.1,2.25,0.17,0.7,0.1,0.1,0.05])
        
        Input
        iparam:     1d array/list which holds the initial guess for each
                     parameter value. The length MUST be 7. Structure:
                     [a, Ex, Ev, S, sig0, Dsig, JnnpEv]
        a:          The area of the entire lineshape
        Ex:         The energy of the uncoupled exciton in eV
        Ev:         The quantum of vibrational energy for the molecule in eV
        S:          The HR factor of the molecule
        sig0:       The broadening of the 0-0 absorption peak in eV 
                     (Usually full width)
        Dsig:       The progressive broadening % accrued for higher 
                     vibrational states
        JnnpEv:     The ratio of the nearest neighbor coupling to the 
                     vibrational quantum
        """
        if not a==-1:
            self.iparam = np.array([a,Ex,Ev,S,sig0,Dsig,JnnpEv])
        else:
            self.iparam = np.array(iparam)
        self.param = np.copy(self.iparam)
    
    def createFitParamBounds(self,bound):
        """
        Function sets the bounds for parameter values during fitting. Note that
        bounds are must NOT be specified for parameters which are frozen. 
        Example:
                which= [True,True,False,True,False,False,False]
                bound=[[0.0, 2.1,        0.1                  ],
                       [1.0, 2.5,        1.0                  ]]
        
        Input
        bound:      2d numpy array holding the paramater bounds to be used during 
                     fitting. If parameters have been frozen by using .freezeFitParam
                     method, then bound will only contain bounds for the parameters
                     which are used during fitting. i.e. bound.shape[1]=nf. 
                     Bound[0] is the lower bound and bound[1] is the upper
                     bound. Note, iparam[i] MUST be in range (bound[0][i],bound[1][i])
                     Structure:
                         [[a_,Ex_,Ev_,S_,sig0_,Dsig_,JnnpEv_],
                          [a^,Ex^,Ev^,S^,sig0^,Dsig^,JnnpEv^]]
        """
        self.bound = bound
    
    def fitFun(self,par):
        """
        Function which is being fit to data. Calls a single vibronic progression
        'lineshape.absFitMu' which takes the functional form explained in
        "derivation.pdf"
        
        Input
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
        
        Output
        res:        1d numpy array holding the simulated absorbance at each
                     provided photon energy.
        """
        p = np.copy(self.param)
        p[self.which] = np.array(par)
        return ls.absLineMu(self.ev,p,4,self.fun)
        
    def fitFunDifference(self,par):
        """
        Function gives the error between fit and data. Used by 
        scipy.optimize.least_squares to minimize the SSE.
        
        Input:
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
                     
        Output
        res:        1d numpy array holding the error between fit and data
        """
        return (self.fitFun(par)[self.fitMask]-self.spec2[self.fitMask])
    
    def plot(self,plotName=''):
        """
        Function gives a plot of the data and fit. Must be called AFTER 
        .initializeFitParams, but can be called before .performFit.
        
        Input
        plotName:   String which titles the plot.
        """
        plt.figure()
        data, = plt.plot(self.ev[self.plotMask],self.spec[self.plotMask],'o',label='Data')
        fit, = plt.plot(self.ev[self.plotMask],
                       (self.ev*ls.absLineMu(self.ev,self.param,4,self.fun))[self.plotMask],label='fit ex')
        fit, = plt.plot(self.ev[self.fitMask],
                       (self.ev*ls.absLineMu(self.ev,self.param,4,self.fun))[self.fitMask],label='fit')
        plt.title(plotName)
        plt.legend()
        plt.show()
        
    def performFit(self,num=7,xtol=3e-16,ftol=1e-10):
        """
        Function modifies param[which] so as to minimize the SSE using
        scipy.optimize.least_squares.
        
        Input
        xtol:       See least_squares documentation
        ftol:       See least_squares documentation
        num:        Integer holding the number of parameters to be printed on
                     each line
        
        Output
        res:        Prints out "Start" iparam[which], "End" param[which] and 
                     "Shift" (param-iparam)[which] as a percentage of upper and
                     lower bounds. This is used to see if any parameters have 
                     "hit" the edges of their range during fitting. This can be
                     seen by as "End" being either 0.0 or 1.0. "Start" can be 
                     used to see if the bounds are too loose, or too strict.
                     And "Shift" gives a sense for how good the initial guess
                     was.
        """
        self.fit = least_squares(self.fitFunDifference,self.iparam[self.which],
                                 verbose=1,bounds=self.bound,xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = np.copy(self.iparam)
            self.param[self.which] = np.copy(self.fit.x)
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = np.copy(self.iparam)
        start = (self.iparam[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        end = (self.param[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        difference = (self.param[self.which]-self.iparam[self.which])/(np.array(self.bound[1])-np.array(self.bound[0]))
        st = lambda x: '{0:6.3f}'.format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames[self.which])%num==0:
            setp = np.arange(len(self.paramNames[self.which])//num)
        else:
            setp = np.arange((len(self.paramNames[self.which])//num)+1)
        for i in setp:
            print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
                             np.vstack([np.vectorize(st2)(self.paramNames[self.which][(num*i):(num*(i+1))]),
                                       np.vectorize(st)(start[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(end[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(difference[(num*i):(num*(i+1))])])
                            ]))   
                                
    def printParam(self,num=7):
        """
        Function prints out the parameter values and names.
        
        Input
        num:        Integer specifying the number of parameters to print onto
                     each line
        """
        st = lambda x: "{0:6.3f}".format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames)%num==0:
            setp = np.arange(len(self.paramNames)//num)
        else:
            setp = np.arange((len(self.paramNames)//num)+1)
        for i in setp:
            print(np.hstack([[[' Name'],['Value']],
                             np.vstack([
                                        np.vectorize(st2)(self.paramNames[(num*i):(num*(i+1))]),
                                        np.vectorize(st)(self.param[(num*i):(num*(i+1))])
                                        ])
                            ]))
class doubleFit:
    """
    This module fits absorption data with as the sum of two vibronic 
    progressions of the form as the last equation of "derivation.pdf" Any of 
    the 14 free parameters can either "frozen" (not varried during gradient 
    descent) or "doubled" (forced to be identical for the two progresions). For
    example, one can "double" S--thus forcing both vibronic progressions to 
    have the same HR factor, or "freeze" only the second vibronic progession's
    intermolecular coupling at 0.
    
    Inputs Required
    ev:         1d Numpy array holding the photon energies of absorption (in eV)
    spec:       1d Numpy array holding the absorption at each photon energy
    fun:        A function "fun(x,mu,sig)" for 'lineshape.absLineMu'. x is 
                 a numpy array holding the domain, mu is the center of the
                 distribution, and sig is the broadening (usually FWHM).
    
    Attributes:
    ev:         1d Numpy array holding the photon energies of absorption (in eV)
    n:          Integer holding the number of absorption data points
    spec:       1d Numpy array holding the absorption at each photon energy
    spec2:      1d Numpy array holding the absorption divided by photon energy
                 (used in fitting proceedure)
    fun:        A function "fun(x,mu,sig)" for 'lineshape.absLineMu'. x is 
                 a numpy array holding the domain, mu is the center of the
                 distribution, and sig is the broadening (usually FWHM).       
    paramNames: 1d Numpy array holding the names of each parameter:
                    a:      The area of the entire lineshape
                    Ex:     The energy of the uncoupled exciton in eV
                    Ev:     The quantum of vibrational energy for the molecule in eV
                    S:      The HR factor of the molecule
                    sig0:   The broadening of the 0-0 absorption peak in eV 
                             (Usually full width)
                    Dsig:   The progressive broadening % accrued for higher 
                             vibrational states
                    JnnpEv: The ratio of the nearest neighbor coupling to the 
                             vibrational quantum
    iparam:     1d numpy array holding the initial guess for paramter values
                 (must be specified using .initializeFitParams BEFORE the fit
                  can be performed). Structure:
                  [a1, Ex1, Ev1, S1, sig01, Dsig1, JnnpEv1, 
                   a2, Ex2, Ev2, S2, sig02, Dsig2, JnnpEv2]
    param:      1d numpy array holding the resulting parameter values after 
                 fitting. Structure:
                  [a1, Ex1, Ev1, S1, sig01, Dsig1, JnnpEv1, 
                   a2, Ex2, Ev2, S2, sig02, Dsig2, JnnpEv2]
    which:      1d bool array holding specifying which of the parameters types
                 will be allowed to varrying during fitting. Note, if any
                 parameter is "frozen," it is fixed for BOTH progessions.
                 Default is to allow all parameters to varry. Can be modified 
                 using .freezeFitParams. Structure:
                  [a?,Ex?,Ev?,S?,sig0?,Dsig?,JnnpEv?]
    doubled:    1d bool array holding specifying which of the parameters will
                 forced to be the same between the two vibronic progresions.
                 Default is that no parameters are restricted to being identical
                 between the two progressions. Can be modified using
                 ".doubleFitParams" Structure:
                  [a?,Ex?,Ev?,S?,sig0?,Dsig?,JnnpEv?]
    bound:      2d numpy array holding the paramater bounds to be used during 
                 fitting. If parameters have been frozen by using .freezeFitParam
                 method, then bound will only contain bounds for the parameters
                 which are used during fitting.  
                 Bound[0] is the lower bound and bound[1] is the upper
                 bound. If paramters have been doubled using .doulbeFitParams,
                 than ONLY the bound for the first progression is used, and
                 the bound on the second progresson's parameter is to be 
                 omited. Example: If S is doubled, only specify S1_,S1^ and
                 leave S2_;S2^ blank. Note, iparam[i] MUST be in range 
                 (bound[0][i],bound[1][i]) Structure when no parameters are 
                 frozen:
                  [[a1_,Ex1_,Ev1_,S1_,sig01_,Dsig1_,JnnpEv1_,
                    a2_,Ex2_,Ev2_,S2_,sig02_,Dsig2_,JnnpEv2_],
                   [a1^,Ex1^,Ev1^,S1^,sig01^,Dsig1^,JnnpEv1^,
                    a2^,Ex2^,Ev2^,S2^,sig02^,Dsig2^,JnnpEv2^]]
    nf:         Value holding the number of parameters allowed to varry
    fitMask:    1d bool array specifying which reflectance data points to be 
                 used during fitting. Default is to use all reflectance data.
                 Can be modified using .createFitRegion
    plotMask:   1d bool array specifying which reflectance data points to be 
                 used during plotting. Default is to use all reflectance data.
                 Can be modified using .createPlotRegion
                 
    
    Best Practice for Use:
          1)  Call singleFit class and provide ev, spec, and fun
         2a)  Specify fit region (.createFitRegion)
          3)  Specify plot region (.createPlotRegion)
    Opt/ 4a)  Freeze parameters NOT used during fitting (.freezeFitParams)
    Opt\ 4b)  Double parameters (.doubleFitParams)
          5)  Provide inital guess for parameter values (.initializeFitParams)
          6)  Set bounds on free fit parameters (.createFitParamBounds)
          7)  Perform Fit (.performFit)
     opt/ 8)  Plot resuts (.plot)
     opt\ 9)  Print fit results (.printParam)
        
    
    """
    def __init__(self,ev,spec,fun = ls.Glorz):
        """
        This method sets up the fitting proceedure
        Input
        ev:         1d Numpy array holding the photon energies of absorption (in eV)
        spec:       1d Numpy array holding the absorption at each photon energy
        fun:        A function "fun(x,mu,sig)" for 'lineshape.absLineMu'. x is 
                     a numpy array holding the domain, mu is the center of the
                     distribution, and sig is the broadening (usually FWHM).
        """
        self.ev = ev
        self.n = ev.shape[0]
        self.spec = spec
        self.spec2 = spec/ev
        self.which = np.full(7,True,dtype='bool')
        self.doubled = np.full(7,False,dtype='bool')
        self.iparam = np.zeros(14)
        self.param = np.copy(self.iparam)
        self.paramNames = np.array(['a1','Ex1','Ev1','S1','sig01','Dsig1','JnnpEv1',
                                    'a2','Ex2','Ev2','S2','sig02','Dsig2','JnnpEv2'])
        self.plotMask = np.full(self.n,True,dtype='bool')
        self.fitMask = np.full(self.n,True,dtype='bool')
        self.fit = 0
        self.bound = 0
        self.nf = 7
        self.fun = fun
        
    def createFitRegion(self,mini,maxi,eV=True):
        """
        Function creates a boolean mask for data to be use during fitting to
        select which data points to fit by. Allows one to select a range of
        either photon energies, or wavelengths inside which to fit.
        
        Input
        mini:       Float: left bound on fit range
        maxi:       Float: right bound on fit range
        ev:         Boolean: when true, fit range is over photon energy (in eV)
                     when false, fit range is selected over wavelength. Note
                     that fits are always assume energy domain, so wavelengths 
                     will later be converted into energies.
        """
        if eV:
            self.fitMask = ls.vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.fitMask = ls.vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
            
    def createPlotRegion(self,mini,maxi,eV=True):
        """
        Function creates a boolean mask for data to be use during plotting to
        select which data points to plot by. Allows one to select a range of
        either photon energies, or wavelengths inside which to fit.
        
        Input
        mini:       Float: left bound on fit range
        maxi:       Float: right bound on fit range
        ev:         Boolean: when true, plot range is over photon energy (in eV)
                     when false, plot range is selected over wavelength. Note
                     that plots are always assume energy domain, so wavelengths
                     will later be converted into energies.
        """
        if eV:
            self.plotMask = ls.vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.plotMask = ls.vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
    
    def freezeFitParams(self,a=True,Ex=True,Ev=True,S=True,sig0=True,Dsig=True,JnnpEv=True):
        """
        Function allows user to freeze particular parameters during fitting. 
        By specifying boolean "False" for a parameter, it is not allowed to vary
        during fitting. Structure: [a?,Ex?,Ev?,S?,sig0?,Dsig?,JnnpEv?]
        
        Input
        which:      1d Boolean list/array. Array MUST be the same length as 
                     iparam/param/paramNames. If which[i]==True, than param[i]
                     is allowed to vary during fitting. If which[i]==False, 
                     than param[i] is frozen during fitting.
        """
        self.which = np.array([a,Ex,Ev,S,sig0,Dsig,JnnpEv],dtype='bool')
        self.nf = 2*np.sum(np.ones(7)[self.which])-np.sum(np.ones(7)[self.doubled])
    
    def doubleFitParams(self,a=False,Ex=False,Ev=False,S=False,sig0=False,Dsig=False,JnnpEv=False):
        """
        Function allows user to "double" particular parameters during fitting. 
        By specifying boolean "True" for a parameter type, the value of that
        parameter is forced to be the same between the two vibronic progressions
        during gradient descent. Note that when a parameter is doubled, 
        progression 2 inherits the value of progession 1, so if progession 2
        is initialized to a different value, that will be overridden at this
        step. Structure: [a?,Ex?,Ev?,S?,sig0?,Dsig?,JnnpEv?]
        
        Input
        double:     1d Boolean list/array. Array MUST be of length  7. If 
                     double[i]==False, than param[i] and param[i+7] will be
                     treated independently during fitting. If double[i]==True, 
                     than param[i+7] is set equal to param[i] durring fitting.
                     If param[i+7]!=param[i] when this function is called, then
                     param[i+7] is overriden.
        """
        self.doubled = np.array([a,Ex,Ev,S,sig0,Dsig,JnnpEv],dtype='bool')
        self.nf = 2*np.sum(np.ones(7)[self.which])-np.sum(np.ones(7)[self.doubled])
        self.iparam[7:14][self.doubled] = self.iparam[0:7][self.doubled]
        self.param[7:14][self.doubled] = self.param[0:7][self.doubled]
    
    def initializeFitParams(self,iparam):
        """
        Function sets the initial guess for parameter values. Note that this 
        function will force self.double to be respected. That is, if 
        double[i]==True, but iparam[i+7]!=iparam[i], then this function will 
        override the value of iparam[i+7] to be iparam[i].
        
        Input
        iparam:     1d array/list which holds the initial guess for each
                     parameter value. The length MUST be 14. Structure:
                     [a1, Ex1, Ev1, S1, sig01, Dsig1, JnnpEv1, 
                      a2, Ex2, Ev2, S2, sig02, Dsig2, JnnpEv2]
        """
        self.iparam = np.array(iparam)
        self.iparam[7:14][self.doubled] = self.iparam[0:7][self.doubled]
        self.param = np.copy(self.iparam)
    
    def createFitParamBounds(self,bound):
        """
        Function sets the bounds for parameter values during fitting. Note that
        bounds must NOT be specified for frozen parameters (in either progression),
        or in the second progression when a parameter is doubled.
        Example:
                which=   [True ,True ,False,True ,False,False,False]
                doubled= [False,False,False,True ,False,False,False]
                bound=  [[ 0.0,  2.1,        0.1,                  
                           0.0,  2.5                               ],
                         [ 1.0,  2.5,        1.0,                  
                           1.0,  2.7                               ]]
        
        Input
        bound:      2d numpy array holding the paramater bounds to be used during 
                     fitting. If parameters have been frozen by using .freezeFitParam
                     method, then bound will only contain bounds for the parameters
                     which are used during fitting. Bound[0] is the lower bound 
                     and bound[1] is the upper bound. If paramters have been 
                     doubled using .doulbeFitParams, than ONLY the bound for 
                     the first progression is used, and the bound on the second 
                     progresson's parameter is to be omited. Example: If S is 
                     doubled, only specify S1_,S1^ and leave S2_;S2^ blank. 
                     Note, iparam[i] MUST be in range (bound[0][i],bound[1][i]) 
                     Structure when no parameters are frozen:
                      [[a1_,Ex1_,Ev1_,S1_,sig01_,Dsig1_,JnnpEv1_,
                        a2_,Ex2_,Ev2_,S2_,sig02_,Dsig2_,JnnpEv2_],
                       [a1^,Ex1^,Ev1^,S1^,sig01^,Dsig1^,JnnpEv1^,
                        a2^,Ex2^,Ev2^,S2^,sig02^,Dsig2^,JnnpEv2^]]
        """
        self.bound = bound
    
    def fitFun(self,par):
        """
        Function which is being fit to data. Calls two vibronic progressions
        'lineshape.absFitMu' which takes the functional form explained in
        "derivation.pdf" Note that fitFun respects both 'which' and 'doubled'
        
        Input
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting (meaning neither param[i] nor param[i+7] 
                     if which[i]=False and not param[i+7] if doubled[i]=True)
        
        Output
        res:        1d numpy array holding the simulated absorbance at each
                     provided photon energy.
        """
        nw = int(np.sum(np.ones(7)[self.which]))
        p = np.copy(self.param)
        p[0:7][self.which] = np.array(par)[0:nw]
        p[7:14][ls.vecAnd([self.which,ls.vecNot(self.doubled)])]=np.array(par)[nw:]
        p[7:14][ls.vecAnd([self.which,self.doubled])]=p[0:7][ls.vecAnd([self.which,self.doubled])]
        return ls.absLineMu(self.ev,p[0:7],4,self.fun)+ls.absLineMu(self.ev,p[7:14],4,self.fun)
        
    def fitFunDifference(self,par):
        """
        Function gives the error between fit and data. Used by 
        scipy.optimize.least_squares to minimize the SSE.
        
        Input:
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting (meaning neither param[i] nor param[i+7] 
                     if which[i]=False and not param[i+7] if doubled[i]=True)
                     
        Output
        res:        1d numpy array holding the error between fit and data
        """
        return (self.fitFun(par)[self.fitMask]-self.spec2[self.fitMask])
    
    def plot(self,plotName=''):
        """
        Function gives a plot of the data and fit. Must be called AFTER 
        .initializeFitParams, but can be called before .performFit.
        
        Input
        plotName:   String which titles the plot.
        """
        plt.figure()
        data, = plt.plot(self.ev[self.plotMask],self.spec[self.plotMask],'o',label='Data')
        fitTot1, = plt.plot(self.ev[self.plotMask],
                       (self.ev*ls.absLineMu(self.ev,self.param[0:7],4,self.fun)+self.ev*ls.absLineMu(self.ev,self.param[7:14],4,self.fun))[self.plotMask],label='Total Fit ex')
        fitTot2, = plt.plot(self.ev[self.fitMask],
                       (self.ev*ls.absLineMu(self.ev,self.param[0:7],4,self.fun)+self.ev*ls.absLineMu(self.ev,self.param[7:14],4,self.fun))[self.fitMask],label='Total Fit')
        fit1, = plt.plot(self.ev[self.fitMask],
                       (self.ev*ls.absLineMu(self.ev,self.param[0:7],4,self.fun))[self.fitMask],label='Fit1')
        fit2, = plt.plot(self.ev[self.fitMask],
                       (self.ev*ls.absLineMu(self.ev,self.param[7:14],4,self.fun))[self.fitMask],label='Fit2')
        plt.title(plotName)
        plt.legend()
        plt.show()
        
    def performFit(self,num=7,xtol=3e-16,ftol=1e-10):
        """
        Function modifies param so as to minimize the SSE using
        scipy.optimize.least_squares while respecting both 'which' and 'doubled'
        
        Input
        xtol:       See least_squares documentation
        ftol:       See least_squares documentation
        num:        Integer holding the number of parameters to be printed on
                     each line
        
        Output
        res:        Prints out "Start" iparam[which], "End" param[which] and 
                     "Shift" (param-iparam)[which] as a percentage of upper and
                     lower bounds. This is used to see if any parameters have 
                     "hit" the edges of their range during fitting. This can be
                     seen by as "End" being either 0.0 or 1.0. "Start" can be 
                     used to see if the bounds are too loose, or too strict.
                     And "Shift" gives a sense for how good the initial guess
                     was.
        """
        mask = np.hstack([self.which,ls.vecAnd([self.which,ls.vecNot(self.doubled)])])
        self.fit = least_squares(self.fitFunDifference,self.iparam[mask],
                                 verbose=1,bounds=self.bound,xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = np.copy(self.iparam)
            self.param[mask] = np.copy(self.fit.x)
            self.param[7:14][ls.vecAnd([self.which,self.doubled])] = np.copy(self.param[0:7][ls.vecAnd([self.which,self.doubled])])
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = np.copy(self.iparam)
        start = (self.iparam[mask]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        end = (self.param[mask]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        difference = (self.param[mask]-self.iparam[mask])/(np.array(self.bound[1])-np.array(self.bound[0]))
        st = lambda x: '{0:6.3f}'.format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames[mask])%num==0:
            setp = np.arange(len(self.paramNames[mask])//num)
        else:
            setp = np.arange((len(self.paramNames[mask])//num)+1)
        for i in setp:
            print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
                             np.vstack([np.vectorize(st2)(self.paramNames[mask][(num*i):(num*(i+1))]),
                                       np.vectorize(st)(start[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(end[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(difference[(num*i):(num*(i+1))])])
                            ]))   
                            
    def printParam(self,num=7):
        """
        Function prints out the parameter values and names.
        
        Input
        num:        Integer specifying the number of parameters to print onto
                     each line
        """
        st = lambda x: "{0:6.3f}".format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames)%num==0:
            setp = np.arange(len(self.paramNames)//num)
        else:
            setp = np.arange((len(self.paramNames)//num)+1)
        for i in setp:
            print(np.hstack([[[' Name'],['Value']],
                             np.vstack([
                                        np.vectorize(st2)(self.paramNames[(num*i):(num*(i+1))]),
                                        np.vectorize(st)(self.param[(num*i):(num*(i+1))])
                                        ])
                            ]))
