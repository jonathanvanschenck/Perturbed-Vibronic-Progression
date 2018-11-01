import numpy as np

def Glorz(x,mu,sig):
    return sig/((2*np.pi)*((x-mu)**2+(sig/2)**2))

def Gauss(x,mu,doubsig):
    sig = doubsig/2
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sig**2))

def vecAnd(listArrays):
    return np.all(np.array(listArrays).T,axis=1)

from math import factorial

def overlap(m,s):
    return np.exp(-s)*(s**m)/factorial(m)

vecOverlap = np.vectorize(overlap)

def rangeNot2(low,high,n):
    res = np.arange(0,high-low+1)+low
    return res[res != n]

def bandshift(m,s,Nmax=10):
    return np.sum(vecOverlap(rangeNot2(0,Nmax,m),s)/(rangeNot2(0,Nmax,m)-m))

def Wcontrib(m,s,W,Nmax=10):
    return (1-0.5*W*bandshift(m,s,Nmax))**2

def intDiscrete(x,y):
    dx=x[1:]-x[:-1]
    dx = np.append(dx,dx[-1])
    return np.sum(dx*y)

def absLineMu2(x,p,n,fun=Glorz):
    #p=area,deltaSig/sig,HR fact,Intermol. Coupling,E00,Evib,sig
    #   0         1         2       3                4   5    6  
    dx = np.absolute(np.mean(x[1:]-x[:-1]))
    x2 = np.arange(p[4]-5*p[5],p[4]+n*p[5]+5*p[5],dx)
    res = overlap(0,p[2])*Wcontrib(0,p[2],4*p[3])*fun(x,p[4]+(2*p[3]*p[6]*overlap(0,p[1])),p[6])
    res2 = overlap(0,p[2])*Wcontrib(0,p[2],4*p[3])*fun(x2,p[4]+(2*p[3]*p[6]*overlap(0,p[1])),p[6])
    for m in np.arange(1,n):
        res = res+overlap(m,p[2])*Wcontrib(m,p[2],4*p[3])*fun(x,p[4]+(2*p[3]*p[6]*overlap(m,p[2]))+m*p[5],(1+m*p[1])*p[6])
        res2 = res2+overlap(m,p[2])*Wcontrib(m,p[2],4*p[3])*fun(x2,p[4]+(2*p[3]*p[6]*overlap(m,p[2]))+m*p[5],(1+m*p[1])*p[6])
    normConst = intDiscrete(x2,res2)
    return p[0]*res/normConst
    
class solutionFit:
    def __init__(self,ev,spec,fun = Glorz):
        self.ev = ev
        self.n = ev.shape[0]
        self.spec = spec
        self.spec2 = spec/ev
        self.which = np.full(7,True,dtype='bool')
        self.iparam = np.zeros(7)
        self.param = np.copy(self.iparam)
        self.paramNames = np.array(['a','Dsig','S','Jnn','E00','Evib','sig'])
        self.plotMask = np.full(self.n,True,dtype='bool')
        self.fitMask = np.full(self.n,True,dtype='bool')
        self.fit = 0
        self.bound = 0
        self.nf = 7
        self.fun = fun
        
    def createFitRegion(self,mini,maxi,eV=True):
        if eV:
            self.fitMask = vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.fitMask = vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
            
    def createPlotRegion(self,mini,maxi,eV=True):
        if eV:
            self.plotMask = vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.plotMask = vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
    
    def freezeFitParams(self,a=True,Dsig=True,S=True,Jnn=True,E00=True,Evib=True,sig=True):
        self.which = np.array([a,Dsig,S,Jnn,E00,Evib,sig],dtype='bool')
        self.nf = np.sum(np.ones(7)[self.which])
    
    def initalizeFitParams(self,iparam=-1,a=-1,Dsig=-1,S=-1,Jnn=-1,E00=-1,Evib=-1,sig=-1):
        if not a==-1:
            self.iparam = np.array([a,Dsig,S,Jnn,E00,Evib,sig])
        else:
            self.iparam = np.array(iparam)
        self.param = np.copy(self.iparam)
    
    def createFitParamBounds(self,bound):
        self.bound = bound
    
    def fitFun(self,par):
        p = np.copy(self.param)
        p[self.which] = np.array(par)
        return absLineMu2(self.ev,p,4,self.fun)
        
    def fitFunDifference(self,par):
        return (self.fitFun(par)[self.fitMask]-self.spec2[self.fitMask])
    
    def plot(self,plotName=''):
        plt.figure()
        data, = plt.plot(self.ev[self.plotMask],self.spec[self.plotMask],'o',label='Data')
        fit, = plt.plot(self.ev[self.plotMask],
                       (self.ev*absLineMu2(self.ev,self.param,4,self.fun))[self.plotMask],label='fit ex')
        fit, = plt.plot(self.ev[self.fitMask],
                       (self.ev*absLineMu2(self.ev,self.param,4,self.fun))[self.fitMask],label='fit')
        plt.title(plotName)
        plt.legend()
        plt.show()
        
    def performFit(self,xtol=3e-16,ftol=1e-10):
        self.fit = least_squares(self.fitFunDifference,self.iparam[self.which],
                                 verbose=1,bounds=self.bound,xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = np.copy(self.iparam)
            self.param[self.which] = np.copy(self.fit.x)
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = np.copy(self.iparam)
        print(np.vectorize(rjust)(np.hstack([np.array([['Name'],['Start'],['End'],['Shift']]),
                         np.vstack([self.paramNames[self.which],
                                    np.round((self.iparam[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0])),2),
                                              np.round((self.param[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0])),2),
                                                        np.round((self.param[self.which]-self.iparam[self.which])/(np.array(self.bound[1])-np.array(self.bound[0])),2)])]),5))

    def printParam(self):
        print(np.vectorize(rjust)(np.hstack([np.array([['Name'],['Value']]),
                         np.vstack([self.paramNames[0:8],
                                    np.round(self.param[0:8],4)
                                    ])]
                                ),6))



