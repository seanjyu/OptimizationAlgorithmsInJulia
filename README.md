# Optimization Algorithms in Julia

The following repo contains implementations of various optimization algorithms in Julia. I created this project to be better understand how these algortihms work and how they could be implemented in code. Therefore the implementations prioritized easier readability and extensibility but at the cost of verbosity and speed.
 <!-- (see comparison tests [here]()).  -->

Many optimization algorithms follow a base pattern and have many variations using this pattern e.g. Nonlinear Conjugate Gradient (Fletcher-Reeves, Dai-Yuan) and Quasi-Newton methods (SR1, DFP). To facillitate this a struct representing the specific variation must be initialized and input to the base algorithm (see [example](#example-usage) below). 

Note some variations are too different from the base pattern despite belonging to same family of algorithms and so are implemented separately e.g. LBFGS or momentum based algorithms (Polyak Heavy Ball and Nesterov Accelerated).

For more production ready functionality please refer to other frameworks such as [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) or [Jump.jl](https://github.com/jump-dev/JuMP.jl).

## Implemented Algorithms
<!-- [Gradient Descent](src/algorithms/GradientDescent.jl) -->
### Unconstrained Optimization 
<summary><a href="src/algorithms/GradientDescent.jl">Gradient Descent</a></summary>
<details>
<summary><a href="src/algorithms/lineSearch">Line Search Methods</a></summary>
    <ul>
        <li><a href="src/algorithms/lineSearch/ArmijoLineSearch.jl"> Armijo Condition</a></li>
        <li><a href="src/algorithms/lineSearch/ArmijoGoldsteinLineSearch.jl">Armijo Goldstein Conditions</a></li>
        <li><a href="src/algorithms/lineSearch/WolfeLineSearch.jl">Wolfe Condition</a></li>
        <li><a href="src/algorithms/lineSearch/MoreThuenteLineSearch.jl">More-Thuente line search</a></li>
    </ul>
</details>
<details>
<summary><a href="src/algorithms/newton">Newton Step Methods</a></summary>
<ul>
    <li><a href="src/algorithms/newton/NewtonMethod.jl">Basic Newton's Method</a></li>
    <li><a href="src/algorithms/newton/QuasiNewtonBase.jl">Quasi-Newton Methods</a></li>
    <ul>
        <li><a href="src/algorithms/newton/SR1.jl">Symmetric Rank 1 (SR1)</a></li>
        <li><a href="src/algorithms/newton/DFP.jl">Davidon-Fletcher-Powell (DFP)</a></li>
        <li><a href="src/algorithms/newton/BFGS.jl">Broyden-Fletcher-Goldfarb-Shanno (BFGS)</a></li>
        <li><a href="src/algorithms/newton/LBFGS.jl">Limited BFGS (LBFGS)</a></li>
    </ul>
<ul>
</details> 
<details>
<summary><a href="src/algorithms/conjugateGradient">Conjugate Gradient Methods</a></summary>
<ul>
    <li><a href="src/algorithms/conjugateGradient/ConjugateGradient.jl">Conjugate Gradient Method</a></li>
    <li><a href="src/algorithms/conjugateGradient/NonlinearCGBase.jl">Non-Linear Conjugate Methods</a></li>
    <ul>
        <li><a href="src/algorithms/conjugateGradient/DaiYuan.jl">Dai-Yuan</a></li>
        <li><a href="src/algorithms/conjugateGradient/FletcherReeves.jl">Fletcher-Reeves</a></li>
        <li><a href="src/algorithms/conjugateGradient/HestenesStiefel.jl">Hestenes-Stiefel</a></li>
        <li><a href="src/algorithms/conjugateGradient/PolakRibiere.jl">Polak-Ribiere</a></li>
    </ul>
<ul>
</details>
<details>
<summary><a href="src/algorithms/momentum">Momentum Methods</a></summary>
<ul>
    <li><a href="src/algorithms/momentum/NesterovAccelerated.jl">Nesterov-Accelerated</a></li>
    <li><a href="src/algorithms/momentum/PolyakHeavyBall.jl">Polyak Heavy Ball</a></li>
</ul>
</details>
<details>
<summary><a href="src/algorithms/trustRegion">Trust-Region Methods</a></summary>
<ul>
    <li><a href="src/algorithms/trustRegion/TrustRegionBase.jl">Base Trust-Region Method</a></li>
    <li><a href="src/algorithms/trustRegion/CauchyPoint.jl">Cauchy-Point</a></li>
    <li><a href="src/algorithms/trustRegion/DogLeg.jl">Dog-Leg</a></li>
    <!-- <li><a href="src/algorithms/conjugateGradient/CGStiefel">CG Stiefel</a></li> -->
</ul>
</details>

### Constrained Optimization
<details>
<summary><a href="src/algorithms/projection">Projection Methods</a></summary>
<ul>
    <li><a href="src/algorithms/projection/ProjectedGradientDescent.jl">Projected Gradient Descent</a></li>
    <li><a href="src/algorithms/projection/POCS.jl">Projection onto Convex Sets (POCS)</a></li>
    <li><a href="src/algorithms/projection/Dykstras.jl">Dykstra's Projection Algorithm</a></li>
</ul>
</details>
<details>
<summary><a href="src/algorithms/penalty">Penalty Methods</a></summary>
<ul>
    <li><a href="src/algorithms/penalty/PenaltyBase.jl">Base Penalty Method</a></li>
    <li><a href="src/algorithms/penalty/L1Penalty.jl">L1 Penalty</a></li>
    <li><a href="src/algorithms/penalty/QuadraticPenalty.jl"> Quadratic Penalty</a></li>
</ul>
</details>
<details>
<summary><a href="src/algorithms/barrier">Barrier Methods</a></summary>
<ul>
    <li><a href="src/algorithms/barrier/BarrierBase.jl"> Base Barrier Method</a></li>
    <li><a href="src/algorithms/barrier/InverseBarrier.jl">Inverse Barrier</a></li>
    <li><a href="src/algorithms/barrier/logBarrier.jl">Log Barrier</a></li>
</ul>
</details>
<details>
<summary><a href="src/algorithms/lagrangian">Lagrangian Methods</a></summary>
<ul>
    <li><a href="src/algorithms/lagrangian/DualAscent.jl">Dual Ascent</a></li>
    <li><a href="src/algorithms/lagrangian/AugmentedLagrangeBase.jl">Base Augmented Lagrangian</a></li>
    <li><a href="src/algorithms/lagrangian/logBarrierALM.jl">Log Barrier Augmented Lagrangian</a></li>
    <li><a href="src/algorithms/lagrangian/ADMM.jl">ADMM</a></li>
    <ul>
        <li><a href="src/algorithms/lagrangian/BasicADMM.jl">Basic ADMM</a></li>
        <li><a href="src/algorithms/lagrangian/LinearizedADMM.jl">Linearized ADMM</a></li>
        <li><a href="src/algorithms/lagrangian/ProximalADMM.jl">Proximal ADMM</a></li>
    </ul>
</ul>
</details>

### Stochastic Optimization
<details>
<summary><a href="src/algorithms/adaptiveStochastic">Adaptive Stochastic Methods</a></summary>
    <ul>
        <li><a href="src/algorithms/adaptiveStochastic/SGD.jl">Stochastic Gradient Descent (SGD)</a></li>
        <ul>
            <li><a href="src/algorithms/adaptiveStochastic/BasicSGD.jl">Basic SGD</a></li>
            <li><a href="src/algorithms/adaptiveStochastic/Adadelta.jl">Adadelta</a></li>
            <li><a href="src/algorithms/adaptiveStochastic/Adagrad.jl">Adagrad</a></li>
            <li><a href="src/algorithms/adaptiveStochastic/RMSprop.jl">RMSProp</a></li>
        </ul>
    </ul>
</details>




## Example Usage
The following is an example using the Dai-Yuan non-linear conjugate gradient method.
```julia
# Define a function
f(x, y) = (10 - x)^2 + 100*(y - x^2)^2
# Create gradient estimator object
GradientEstimator = FiniteDifferenceMultivariate{5,:central}(1e-8)
# Create convergence object
c = ConvergenceCriteria(gradTol = 1e-6)
# Create step length strategy
stepLengthStrategy = Backtracking(0.8)
# Create line search strategy
wolfeLineSearch = WolfeLineSearch(c1 = 1e-4, c2 = 0.95, stepLengthStrategy = stepLengthStrategy, curvTol = 1e-16)
# Create non-linear conjugate method
daiYuan = DaiYuan()
# Create starting point
x0 = [0, 0]
# Call base non-linear conjugate gradient method
result = NonlinearCGOpt(f, 
                        x0, 
                        GradientEstimator, 
                        daiYuan, 
                        wolfeLineSearch, 
                        c, 
                        alpha = 2, 
                        lim = 1000, 
                        lineSearchLim = 70, 
                        tol = 1e-5, 
                        printIter = true)
# Read results
minCoord = result.minimumPoint
        
```

Note that utility functions/structs like `convergence` and `gradientEstimator` were also implemented. 
<!-- ## Installation -->

<!-- ## Documentation
Each  -->


<!-- ## Usage -->
