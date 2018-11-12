# PerturbedVibronicProgression
Provides curving fitting for perturbed vibronic progressions. See ExampleScript.py and ExampleData.csv to see this fitting in practice. A fuller explanation can be found in "derivation.pdf"

Best Practice for 'fit.singleFit'

           1)  Call singleFit class and provide ev, spec, and fun
           
           2)  Specify fit region (.createFitRegion)
           
           3)  Specify plot region (.createPlotRegion)
           
      Opt  4)  Freeze parameters NOT used during fitting (.freezeFitParams)
      
           5)  Provide inital guess for parameter values (.initializeFitParams)
           
           6)  Set bounds on free fit parameters (.createFitParamBounds)
           
           7)  Perform Fit (.performFit)
           
      opt/ 8)  Plot resuts (.plot)
      
      opt\ 9)  Print fit results (.printParam)


Best Practice for 'fit.doubleFit'

           1)  Call singleFit class and provide ev, spec, and fun
           
           2)  Specify fit region (.createFitRegion)
           
           3)  Specify plot region (.createPlotRegion)
           
     Opt/ 4a)  Freeze parameters NOT used during fitting (.freezeFitParams)
     
     Opt\ 4b)  Double parameters (.doubleFitParams)
     
           5)  Provide inital guess for parameter values (.initializeFitParams)
           
           6)  Set bounds on free fit parameters (.createFitParamBounds)
           
           7)  Perform Fit (.performFit)
           
      opt/ 8)  Plot resuts (.plot)
      
      opt\ 9)  Print fit results (.printParam)
