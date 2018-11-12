# -*- coding: utf-8 -*-
#%%Set Path to folder containing ExampleData.csv and fit.py
loc = 'Path/to/folder'
#%%Imports
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(loc))
import fit
#Pull Spectrum
xylCB = np.genfromtxt(loc+'/XylInCBAbs2.csv',
                      skip_header=1,delimiter=',')
#%%Fit Single
fitTest = fit.singleFit(1240/xylCB[:,0],xylCB[:,4])
fitTest.createFitRegion(1.6,2.4)
fitTest.createPlotRegion(1.5,2.5)
#fitTest.initializeFitParams([.1,0,0.8,0.0,1.85,.16,0.08])
fitTest.initializeFitParams([.02,1.87,.18,0.7,.16,0.0,0.0])
fitTest.freezeFitParams(JnnpEv=False,Dsig=False)
#                               a    Dsig    S      Jnn      E00     Evib     sig
#fitTest.createFitParamBounds(((.01,          0.1,            1.8,     .15,    .01),
#                              (.5,           1.0,            2.0,     .22,    .15)))
#                               a    Ex,    Ev,     S,     sig0,     Dsig,     JnnpEv
fitTest.createFitParamBounds(((.01,  1.8,   .15,   0.1,    .01                        ),
                              (.5,   2.0,   .22,   1.0,    .25                        )))
fitTest.performFit()
fitTest.plot()
fitTest.printParam()
#%%Fit Double
fitTest = fit.doubleFit(1240/xylCB[:,0],xylCB[:,4])
fitTest.createFitRegion(1.6,2.4)
fitTest.createPlotRegion(1.5,2.5)
fitTest.initializeFitParams(np.array([.1,1.82,.16,0.6,0.08,0,0.0,
                                     .1,1.87,.16,0.6,0.08,0,0.0]))
fitTest.freezeFitParams(JnnpEv=False,Dsig=False)
#                               a    Ex,    Ev,    S,   sig0,  Dsig,  JnnpEv
fitTest.createFitParamBounds(((.01,  1.8,  0.15,  0.1,  .01,                 
                               .01,  1.8,  0.15,  0.1,  .01                  ),
                              (.5 ,  2.0,  0.22,  1.0,  .15,
                               .5 ,  2.0,  0.22,  1.0,  .15                  )))
fitTest.performFit()
fitTest.plot()
fitTest.printParam()
#%%Fit Double, but force S and Ev to be the same
fitTest = fit.doubleFit(1240/xylCB[:,0],xylCB[:,4])
fitTest.createFitRegion(1.6,2.4)
fitTest.createPlotRegion(1.5,2.5)
fitTest.initializeFitParams(np.array([.01,1.82,.17,0.6,0.08,0,0.0,
                                     .01,1.87,.17,0.6,0.08,0,0.0]))
fitTest.freezeFitParams(JnnpEv=False,Dsig=False)
fitTest.doubleFitParams(S=True,Ev=True)
#                               a    Ex,    Ev,    S,   sig0,  Dsig,  JnnpEv
fitTest.createFitParamBounds(((.001, 1.8,  0.15,  0.1,  .01,                 
                               .001, 1.8,               .01                  ),
                              (.5 ,  2.0,  0.22,  1.0,  .15,
                               .5 ,  2.0,               .15                  )))
fitTest.performFit()
fitTest.plot()
fitTest.printParam()
#%%Fit Double, but also force sig0 to be the same
fitTest = fit.doubleFit(1240/xylCB[:,0],xylCB[:,4])
fitTest.createFitRegion(1.6,2.4)
fitTest.createPlotRegion(1.5,2.5)
fitTest.initializeFitParams(np.array([.01,1.82,.17,0.6,0.08,0,0.0,
                                     .01,1.87,.17,0.6,0.08,0,0.0]))
fitTest.freezeFitParams(JnnpEv=False,Dsig=False)
fitTest.doubleFitParams(S=True,Ev=True,sig0=True)
#                               a    Ex,    Ev,    S,   sig0,  Dsig,  JnnpEv
fitTest.createFitParamBounds(((.001, 1.8,  0.15,  0.1,  .01,                 
                               .001, 1.8,                                    ),
                              (.5 ,  2.0,  0.22,  1.0,  .15,
                               .5 ,  2.0,                                    )))
fitTest.performFit()
fitTest.plot()
fitTest.printParam()
#%%Fit Double, now allow Dsig to varry, but force it to be doubled
fitTest = fit.doubleFit(1240/xylCB[:,0],xylCB[:,4])
fitTest.createFitRegion(1.7,2.4)
fitTest.createPlotRegion(1.5,2.5)
fitTest.initializeFitParams(np.array([.01,1.82,.17,0.6,0.08,0.1,0.0,
                                     .01,1.87,.17,0.6,0.08,0.1,0.0]))
fitTest.freezeFitParams(JnnpEv=False)
fitTest.doubleFitParams(S=True,Ev=True,sig0=True,Dsig=True)
#                               a    Ex,    Ev,    S,   sig0,  Dsig,  JnnpEv
fitTest.createFitParamBounds(((.001, 1.8,  0.15,  0.1,  .01,   0.0,          
                               .001, 1.8                                     ),
                              (.5 ,  2.0,  0.22,  1.0,  .15,   1.0,
                               .5 ,  2.0,                                    )))
fitTest.performFit()
fitTest.plot()
fitTest.printParam()
#%%Fit Double, set Ev, becuase the previous block doesn't fit quite right
fitTest = fit.doubleFit(1240/xylCB[:,0],xylCB[:,4])
fitTest.createFitRegion(1.5,2.4)
fitTest.createPlotRegion(1.5,2.5)
fitTest.initializeFitParams(np.array([.01,1.82,.175,0.6,0.06,0.1,0.0,
                                     .01,1.87,.175,0.6,0.06,0.1,0.0]))
fitTest.freezeFitParams(JnnpEv=False,Ev=False)
fitTest.doubleFitParams(S=True,Ev=True,sig0=True,Dsig=True)
#                               a    Ex,    Ev,    S,   sig0,  Dsig,  JnnpEv
fitTest.createFitParamBounds(((.001, 1.8,         0.1,  .01,   0.0,          
                               .001, 1.8                                     ),
                              (.5 ,  2.0,         1.0,  .15,   1.0,
                               .5 ,  2.0,                                    )))
fitTest.performFit()
fitTest.plot()
fitTest.printParam()
