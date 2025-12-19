# Utilities

## Logger
### Struct
### Methods

## Convergence Criteria
The following convergence criteria were implemented:
- Gradient tolerance
    - Gradient norm to be greater than tolerance .
- X tolerance
    - Change in norm coordinate value over an iteration to be greater than tolerance value.
- Function value
    - Change in absolute function value over an iteration to be greater than tolerance value.
- Step size tolerance
    - Change in step size over an iteration to be greater than tolerance value.

Note the default criteria is a gradient tolerance of 1e-6.
<!-- Create struct containing tolerance for each criteria to check   -->
### Structs/Type
ConvergenceCriteria - stores tolerances and convergence values. 
- Inputs
    - gradTol (float64) - 
    - xTol (float64) -
    - fTol (float64) - 
    - stepTol (float64) - 
    - all (boolean) - 
    - reason (boolean) - 

### Methods
Check Convergence -  function to check convergence. Returns boolean value representing whether or not the method converged and a string for convergence 
```
converged, reason = CheckConvergence(c::ConvergenceCriteria, grad, xCur, xOld, fCur, fOld, step)
```

### Usage
```Julia
# create the convergence criteria struct
c = ConvergenceCriteria()
```