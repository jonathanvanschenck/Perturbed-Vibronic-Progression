# -*- coding: utf-8 -*-
"""
Created on 11/8/18 Updated:
Written By: Jonathan D B Van Schenck

This module holds a series of special functions used during fitting. The 
lineshape ("absLineMu") is the result of using first order perturbation theory 
on a 1D molecular crystalline chain for nearest neighbor interactions allowing 
for excitonic and vibrational degrees of freedom. The 7 free parameters used 
by "absLineMu" are:
    a:      The area of the entire lineshape
    Ex:     The energy of the uncoupled exciton in eV
    Ev:     The quantum of vibrational energy for the molecule in eV
    S:      The HR factor of the molecule
    sig0:   The broadening of the 0-0 absorption peak in eV (Usually full width)
    Dsig:   The progressive broadening % accrued for higher vibrational states
    JnnpEv: The ratio of the nearest neighbor coupling to the vibrational quantum
The derivations behind the functional forms of each of these function can be 
seen in this directory under "derivation.pdf"

Dependancies
numpy:          Used for array handling
math.factorial: Calculates the factorial

Functions
vecAnd:         A vectorized implementation of "and"
vecNot:         A vectorized implementation of "not"
Glorz:          Normalized lorentzian function
Gauss:          Implimentation of gaussian which takes 2*sig as broadening
                 input, so as to more closely match the FWHM used by Glorz
overlap:        Calculates the special case Frank-Condon overlap integral
                 |<0,vt>|**2 for HR factor = S
vecOverlap:     A vectorized implementation of "overlap"
reangeNot2:     Returns the integers between low and high sans the value n
bandshift:      Returns the contribution of |Psi^(1)> to the transition dipole
                 operator matrix element <G|mu|Psi> for 1-ps (See derivation.pdf)
Wcontrib:       Returns the approximate value of |<G|mu|Psi>|**2 for 1-ps to
                 first order in Jnn/Ev
Wcontrib2:      A depricated version of Wcontrib with uses the exicton bandwidth
                 instead of the nearest neighbot coupling.
intDiscrete:    Calculates the Reimann sum of {x,y} for normalization
absLineMu:      Returns the absorption lineshape for the system described in
                 "derivation.pdf"
absLineMu2:     A depricated version of absLineMu which has parameters listed
                 in a different order.
"""
import numpy as np
from math import factorial

def vecAnd(listArrays):
    """
    This function applies "and" elementwise across a list of numpy arrays. It 
    is used primarially to create plotting/fitting masks by combining boolean
    arrays.
    
    Input
    listArrays:     A list of 1d bool arrays, formatted where each row is holds
                     the elements of a boolean array to be and'ed:
                     [[bool array #1],[bool array #2],[bool array #3]]
                     
    Output
    res:            A 1-d numpy bool array where each element is the total and
                     of all the corresponding elements in "listArrays":
                     [[#1][0] and [#2][0] and ..., [#1][1] and [#2][1] and ..., ...]
    """
    return np.all(np.array(listArrays).T,axis=1)
    
def vecNot(boolVec):
    """
    This function applies "not" elementwise across a 1d boolean numpy array.
    
    Input
    boolVec:        1d bool Array to be "not"ed
                     
    Output
    res:            A 1-d numpy bool array where each element is the logical
                     negation of each element in boolVec
    """
    return np.vectorize(lambda x: not x)(boolVec)
    
def Glorz(x,mu,sig):
    """
    This function implements a normalized lorentzian function
    
    Inputs
    x:          Numpy array holding the domain Glorz is to be applied over
    mu:         The center of the distribution
    sig:        The FWHM
    
    Outputs
    res:        Numpy array holding the y values of Glorz
    """
    return sig/((2*np.pi)*((x-mu)**2+(sig/2)**2))

def Gauss(x,mu,doubsig):
    """
    This function implements a normalized gaussian function, which has broadening
    close to the FWHM, rather than the standard deviation
    
    Inputs
    x:          Numpy array holding the domain Gauss is to be applied over
    mu:         The center of the distribution
    doubsig:    2*standardDeviation of the distribtuion (notice doubsig~FWHM)
    
    Outputs
    res:        Numpy array holding the y values of Gauss
    """
    sig = doubsig/2
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sig**2))
    
def overlap(m,s):
    """
    This function calculates the square norm of the overlap of a harmonic 
    oscillator wavefunction and another HO wavefunction shifted by sqrt(2*s) 
    with respect to the other in the special case where one state is ground and
    the other state is in the mth state. This is called a Frank Condon 
    coefficient. See "derivation.pdf"
    
    Inputs
    m:          Integer specifying the quanta of vibration in the shifted HO
                 state
    s:          Float representing the HR factor of the molecule
    
    Outputs
    res:        Frank Condon coefficent: |<0|vt>|**2
    """
    return np.exp(-s)*(s**m)/factorial(m)
    
vecOverlap = np.vectorize(overlap)

def rangeNot2(low,high,n):
    """
    This function returns a list of integers between low and high that excludes
    the integer n, should n be contained within (low,high). Used to calculate
    the sum in "bandshift." See "derivation.pdf"
    
    Inputs
    low:        Integer specifying the lower bound (inclusive) of the range
    high:       Integer specifying the upper bound (inclusive) of the range
    n:          Integer which is to be excluded from the returned array if it
                 is contained within (low,high).
    
    Outputs
    res:        Array of integers (actually floats) between low and high 
                 (inclusive) which excludes the integer n.
    """
    res = np.arange(0,high-low+1)+low
    return res[res != n]
    
def bandshift(m,s,Nmax=10):
    """
    This function calculates the contribution of |Psi^(1)> to <G|mu|Psi> for 
    1-particle states. See "derivation.pdf"
    
    Inputs
    m:          Integer specifying the vibrational quanta of the unperturbed
                 1-particle state |Psi^(0)>
    s:          Float holding the HR factor of the molecule
    Nmax:       Integer specifying the upper bound of the sum in "derivation.pdf"
                 must be much larger than m for good convergence.
    
    Outputs
    res:        Value of the sum in "derivation.pdf"
    """
    return np.sum(vecOverlap(rangeNot2(0,Nmax,m),s)/(rangeNot2(0,Nmax,m)-m))
    
def Wcontrib(m,s,JnnpEv,Nmax=10):
    """
    This function approximates the value of |<G|mu|Psi>|^2 for 1-particle
    states to first order in Jnn/Ev. See "derivation.pdf"
    
    Inputs
    m:          Integer specifying the vibrational quanta of the unperturbed
                 1-particle state |Psi^(0)>
    s:          Float holding the HR factor of the molecule
    JnnpEv:     Ratio of the nearest neighbor coupling to the vibrational
                 energy quantum.
    Nmax:       Integer specifying the upper bound of the sum in "derivation.pdf"
                 must be much larger than m for good convergence.
    
    Outputs
    res:        Value of |<G|mu|Psi>|^2 for 1-particle states
    """
    return (1-2*JnnpEv*bandshift(m,s,Nmax))**2
    
def Wcontrib2(m,s,W,Nmax=10):
    """
    *This is a depricated version of 'Wcontrib'*
    
    This function approximates the value of |<G|mu|Psi>|^2 for 1-particle
    states to first order in Jnn/Ev. See "derivation.pdf"
    
    Inputs
    m:          Integer specifying the vibrational quanta of the unperturbed
                 1-particle state |Psi^(0)>
    s:          Float holding the HR factor of the molecule
    W:          Float holding the ratio of the exciton bandwidth to the 
                 vibrational energy quantum. Assumes 2 nearest neighbors
    Nmax:       Integer specifying the upper bound of the sum in "derivation.pdf"
                 must be much larger than m for good convergence.
    
    Outputs
    res:        Value of |<G|mu|Psi>|^2 for 1-particle states
    """
    return (1-0.5*W*bandshift(m,s,Nmax))**2
    
def intDiscrete(x,y):
    """
    This function calculates the Reimann sum of {x,y}. Used primarially for 
    normalization.
    
    Inputs
    x:          1d Numpy array holding x values to be summed over. MUST have 
                 the same length as y.
    y:          1d Numpy array holding y values to be summed over. MUST have
                 the same length as x.
    
    Outputs
    res:        Value of the Reimann sum of {x,y}
    """
    dx=x[1:]-x[:-1]
    dx = np.append(dx,dx[-1])
    return np.sum(dx*y)
    
def absLineMu(x,p,n,fun=Glorz):
    """
    This function returns the approximate absorption for a 1d ring of nearest
    neighbor-coupled molecules which can host both an exciton and vibrational
    quanta. This implementation *almost* uses the final equation in 
    "derivation.pdf," with the exception that the entire lineshape is first
    normalized, and then scaled by a. The parameters used are as follows:
        a:      The area of the entire lineshape
        Ex:     The energy of the uncoupled exciton in eV
        Ev:     The quantum of vibrational energy for the molecule in eV
        S:      The HR factor of the molecule
        sig0:   The broadening of the 0-0 absorption peak in eV (Usually full width)
        Dsig:   The progressive broadening % accrued for higher vibrational states
        JnnpEv: The ratio of the nearest neighbor coupling to the vibrational quantum
    
    Inputs
    x:          1d Numpy array holding the photon energies (in eV) for which
                 the absorption is to be calculated for.
    p:          1d Numpy array holding parameters values. Structure:
                 [a,Ex,Ev,S,sig0,Dsig,JnnpEv]
    fun:        A function holding the lineshape of each constituent transition
                 in the absorption to be combined together. It MUST have the 
                 form: fun(x,mu,sig) where mu is the center transition energy
                 and sig is the width of the transistion (usually FW).
    
    Outputs
    res:        1d Numpy array holding the absorbance/(photon eV) for each
                 photon energy specified by x.
    """
    #p=area, Ex , Ev , S , sig0 ,Dsig , Jnn/Ev
    #   0    1    2    3     4    5       6  
    dx = np.absolute(np.mean(x[1:]-x[:-1]))
    x2 = np.arange(p[1]-5*p[2],p[1]+(n+5)*p[5],dx)
    res = overlap(0,p[3])*Wcontrib(0,p[3],p[6])*fun(x,p[1]+p[2]*(0+2*p[6]*overlap(0,p[3])),p[4]*(1+0*p[5]))
    res2 = overlap(0,p[3])*Wcontrib(0,p[3],p[6])*fun(x2,p[1]+p[2]*(0+2*p[6]*overlap(0,p[3])),p[4]*(1+0*p[5]))
    for m in np.arange(1,n):
        res += overlap(m,p[3])*Wcontrib(m,p[3],p[6])*fun(x,p[1]+p[2]*(m+2*p[6]*overlap(m,p[3])),p[4]*(1+m*p[5]))
        res2 += overlap(m,p[3])*Wcontrib(m,p[3],p[6])*fun(x2,p[1]+p[2]*(m+2*p[6]*overlap(m,p[3])),p[4]*(1+m*p[5]))
    normConst = intDiscrete(x2,res2)
    return p[0]*res/normConst
    
def absLineMu2(x,p,n,fun=Glorz):
    """
    *This is a depricated version of 'absLineMu'*
    
    This function returns the approximate absorption for a 1d ring of nearest
    neighbor-coupled molecules which can host both an exciton and vibrational
    quanta. This implementation *almost* uses the final equation in 
    "derivation.pdf," with the exception that the entire lineshape is first
    normalized, and then scaled by a. The parameters used are as follows:
        a:      The area of the entire lineshape
        Dsig:   The progressive broadening % accrued for higher vibrational states
        S:      The HR factor of the molecule
        JnnpEv: The ratio of the nearest neighbor coupling to the vibrational quantum
        Ex:     The energy of the uncoupled exciton in eV
        Ev:     The quantum of vibrational energy for the molecule in eV
        sig0:   The broadening of the 0-0 absorption peak in eV (Usually full width)
        
    Inputs
    x:          1d Numpy array holding the photon energies (in eV) for which
                 the absorption is to be calculated for.
    p:          1d Numpy array holding parameters values. Structure:
                 [a,Dsig,S,JnnpEv,Ex,Ev,sig0]
    fun:        A function holding the lineshape of each constituent transition
                 in the absorption to be combined together. It MUST have the 
                 form: fun(x,mu,sig) where mu is the center transition energy
                 and sig is the width of the transistion (usually FW).
    
    Outputs
    res:        1d Numpy array holding the absorbance/(photon eV) for each
                 photon energy specified by x.
    """
    #p=area,deltaSig/sig,HR fact,Intermol. Coupling,E00,Evib,sig
    #   0         1         2       3                4   5    6  
    dx = np.absolute(np.mean(x[1:]-x[:-1]))
    x2 = np.arange(p[4]-5*p[5],p[4]+n*p[5]+5*p[5],dx)
    res = overlap(0,p[2])*Wcontrib2(0,p[2],4*p[3])*fun(x,p[4]+(2*p[3]*p[6]*overlap(0,p[1])),p[6])
    res2 = overlap(0,p[2])*Wcontrib2(0,p[2],4*p[3])*fun(x2,p[4]+(2*p[3]*p[6]*overlap(0,p[1])),p[6])
    for m in np.arange(1,n):
        res = res+overlap(m,p[2])*Wcontrib2(m,p[2],4*p[3])*fun(x,p[4]+(2*p[3]*p[6]*overlap(m,p[2]))+m*p[5],(1+m*p[1])*p[6])
        res2 = res2+overlap(m,p[2])*Wcontrib2(m,p[2],4*p[3])*fun(x2,p[4]+(2*p[3]*p[6]*overlap(m,p[2]))+m*p[5],(1+m*p[1])*p[6])
    normConst = intDiscrete(x2,res2)
    return p[0]*res/normConst