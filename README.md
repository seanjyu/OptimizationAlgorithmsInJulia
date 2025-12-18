# Optimization Algorithms in Julia

The following repo contains implementations of various optimization algorithms in Julia. I created this project to be better understand how these algortihms work and how they could be implemented in code. Therefore the implementations prioritized easier readability and extensibility but at the cost of verbosity and speed (see comparison tests [here]()). 

Many optimization algorithms follow a base pattern and have many variations using this pattern e.g. Nonlinear Conjugate Gradient (Fletcher-Reeves, Dai-Yuan) and Quasi-Newton methods (SR1, DFP). To facillitate this a struct representing the specific variation must be initialized and input to the base algorithm (see [example](#example-usage) below). 

Note some variations are too different from the base pattern despite belonging to same family of algorithms and so are implemented separately e.g. LBFGS or momentum based algorithms (Polyak Heavy Ball and Nesterov Accelerated).

For more production ready functionality please refer to other frameworks such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) or [Jump.jl](https://github.com/jump-dev/JuMP.jl).

## Algorithms Implemented
- [Gradient Descent](src\algorithms\GradientDescent.jl)
- [Line Search Methods](src\algorithms\lineSearch)
    - Armijo Condition
    - Armijo Goldstein Conditions
    - Wolfe Condition
    - More-Thuente line search

- Newton Step Methods
    - Basic Newton's Method
    - Quasi-Newton Methods
        - Symmetric Rank 1 (SR1)
        - Davidon-Fletcher-Powell (DFP)
        - Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - Limited BFGS (LBFGS)

## Example Usage
The following is an example 

Note that utility functions/structs like `convergence` and `gradientEstimator` were also implemented. 
## Installation

## Documentation
Each 


<!-- ## Usage -->
