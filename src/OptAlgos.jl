module OptAlgos 
"""

"""

using LinearAlgebra
using ForwardDiff
using ReverseDiff
using Statistics
using Random
using Printf

# ==============================
# Exports 
# ==============================
# -------- Utilities --------
# Gradient Estimation
export  GradientEstimator,
        gradient,
        hessian

# Gradient Estimators
export  UnivariateForwardDifference, 
        MultivariateForwardDifference,
        ForwardDiffEstimator, ReverseDiffEstimator
export FiniteDifferenceUnivariate, FiniteDifferenceMultivariate
export MiniBatchGradientEstimator
# Convergence Criteria
export ConvergenceCriteria, CheckConvergence

# Logger
export NoLogger, Logger, NoAlgorithmData, AlgorithmData, initLogger, logIter!, finalizeLogger!, setConvergenceReason!
# Test Functions
export  UnivariateQuadraticFunction, discriminant, roots,
        MultivariateQuadraticFunction, eigenvalues, 
        # Shared methods
        trueGradient, trueMinimum
        
# Unconstrained Optimization Wrapper
export UnconstrainedOptMethod, solveUnconstrainedOpt

# Models
export  Model, initModel,
        LinearRegressionModel 

# Loss functions
export LossFunction, computeLoss, modelLossFunction, MSE

# Constraint Functions
export Constraint, project, isFeasible, violation, gradient, BoxConstraint, CompositeConstraint, ProjectionAlgorithm, LinearEqualityConstraint, LinearInequalityConstraint

# -------- Algorithms --------
export GradientDescent

# Step length methods
export calculateStepLength, Backtracking, QuadraticInterpolation, StepLengthStrategy

# Line search methods
export LineSearchMethod, stepSearch, lineSearch, ArmijoLineSearch, ArmijoGoldsteinLineSearch, WolfeLineSearch, MoreThuenteLineSearch

# Newton/Quasi Newton methods
export NewtonMethod, QuasiNewtonMethod, QuasiNewtonOpt, BFGSH, BFGSB, SR1Module, SR1H, SR1B, DFPH, DFPB, updateApproximation, LBFGS

# Conjugate Gradient and Nonlinear Conjugate Gradient Methods
export NonlinearCGMethod, NonlinearCGOpt, DaiYuan, FletcherReeves, HestenesStiefel, PolakRibiere

#
export ProjectedGradientDescent, Dykstra, POCS

# Unconstrained Optimization Wrappers
export solveUnconstrainedOpt, UOWNewtonOpt, UOWQuasiNewtonOpt 

# ============================= 
# Include files 
# =============================

# ======== Utilities ========
# Gradient Estimation
include("utils/GradientEstimators/GradientEstimatorInterface.jl")
include("utils/GradientEstimators/AutomaticDifferentiation.jl")
include("utils/GradientEstimators/ForwardDifference.jl")
include("utils/GradientEstimators/JuliaAutoDiff.jl")
include("utils/GradientEstimators/FiniteDifference.jl")
include("utils/GradientEstimators/StochasticGradientEstimator.jl")

# Convergence Criteria
include("utils/ConvergenceCriteria.jl")

# Constraint Functions
include("utils/ConstraintFunctions/ConstraintInterface.jl")
include("utils/ConstraintFunctions/LinearEqualityConstraint.jl")
include("utils/ConstraintFunctions/LinearInequalityConstraint.jl")
include("utils/ConstraintFunctions/BoxConstraint.jl")
include("utils/ConstraintFunctions/CompositeConstraint.jl")

# Models
include("utils/Models/modelInterface.jl")
include("utils/Models/linearRegression.jl")

# Loss Functions (for model parameter optimization)
include("utils//LossFunctions/LossFunctionInterface.jl")
include("utils/LossFunctions/MSE.jl")

# Logging
include("utils/Logger.jl")

# Test Functions
include("utils/TestFunctions.jl")

# ======== Algorithms ========
# gradient descent
include("algorithms/GradientDescent.jl")

# line search
include("algorithms/lineSearch/StepLengthStrategy/StepLengthStrategy.jl")
include("algorithms/lineSearch/StepLengthStrategy/Backtracking.jl")
include("algorithms/lineSearch/StepLengthStrategy/QuadraticInterpolation.jl")

include("algorithms/lineSearch/lineSearchBase.jl")
include("algorithms/lineSearch/ArmijoLineSearch.jl")
include("algorithms/lineSearch/ArmijoGoldsteinLineSearch.jl")
include("algorithms/lineSearch/WolfeLineSearch.jl")
include("algorithms/lineSearch/MoreThuenteLineSearch.jl")

# Newton step algorithms
include("algorithms/newton/NewtonMethod.jl")
include("algorithms/newton/QuasiNewtonBase.jl")
include("algorithms/newton/BFGS.jl")
include("algorithms/newton/DFP.jl")
include("algorithms/newton/SR1.jl")
include("algorithms/newton/LBFGS.jl")

# Momentum methods
include("algorithms/momentum/PolyakHeavyBall.jl")
include("algorithms/momentum/NesterovAccelerated.jl")

# Conjugate Gradient algorithms
# include("algorithms/conjugateGradient/ConjugateGradient.jl")
include("algorithms/conjugateGradient/NonlinearCGBase.jl")
include("algorithms/conjugateGradient/DaiYuan.jl")
include("algorithms/conjugateGradient/FletcherReeves.jl")
include("algorithms/conjugateGradient/HestenesStiefel.jl")
include("algorithms/conjugateGradient/PolakRibiere.jl")

# Trust Region Algorithms
# include("algorithms/trustRegion/TrustRegionBase.jl")
# include("algorithms/trustRegion/CauchyPoint.jl")
# include("algorithms/trustRegion/DogLeg.jl")

# Adaptive Stochastic Algorithms
include("algorithms/adaptiveStochastic/SGD.jl")
include("algorithms/adaptiveStochastic/VanillaSGD.jl")
include("algorithms/adaptiveStochastic/Adadelta.jl")
include("algorithms/adaptiveStochastic/Adagrad.jl")
include("algorithms/adaptiveStochastic/Adam.jl")
include("algorithms/adaptiveStochastic/RMSprop.jl")

# Projected Gradient Descent
include("algorithms/projection/ProjectedGradientDescent.jl")
include("algorithms/projection/Dykstras.jl")
include("algorithms/projection/POCS.jl")

# ======== Unconstrained Optimization Wrappers ========
include("utils/UnconstrainedOptWrapper/UnconstrainedOptSolverInterface.jl")
include("utils/UnconstrainedOptWrapper/NewtonOptSolver.jl")
include("utils/UnconstrainedOptWrapper/QuasiNewtonOptSolver.jl")


end