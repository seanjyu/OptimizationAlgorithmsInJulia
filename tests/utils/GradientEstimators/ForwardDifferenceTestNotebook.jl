### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ e7d4d847-8215-4cb2-879c-85ef9707d311
begin
	include("../../utils/GradientEstimators/ForwardDifference.jl")
	include("../../utils/TestFunctions.jl")
	using Test
	using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
	using .UnivariateForwardDifferenceModule, .MultivariateForwardDifferenceModule
end

# ╔═╡ 5f696645-1fad-4030-96d6-773f7f8c7201
md""" 
# Finite Difference Testing
## Univariate Finite Difference
Import Functions
"""

# ╔═╡ 5e36d0ac-d96f-40d3-ba22-4350f3dfd10b
md"""
Create test univariate quadratic function

$x^{2} + 2x + 3$

"""

# ╔═╡ 4eaa3e22-6aa5-4dfc-bd1e-da27356c0412
begin
	quadraticFunction = UnivariateQuadraticFunctionModule.UnivariateQuadraticFunction(1, 2, 3)
	testGradient = UnivariateQuadraticFunctionModule.trueGradient(quadraticFunction, 1)
	println("Gradient at 1: ", testGradient)
end

# ╔═╡ 083bf251-9fad-40de-89cb-4a2e67cba990
md"""
Test gradient at various points, note the default h is 1e-5 so the tolerance was set to 1e-4
"""

# ╔═╡ 4b26410d-00ad-4810-96dc-28c921e4fb50
@testset "Test Univariate Forward Difference" begin
	@test isapprox(testGradient, UnivariateForwardDifferenceModule.UnivariateForwardDifference(quadraticFunction, 1); atol=1e-4)
	@test isapprox(UnivariateQuadraticFunctionModule.trueGradient(quadraticFunction, 2), UnivariateForwardDifferenceModule.UnivariateForwardDifference(quadraticFunction, 2); atol=1e-4)
	@test isapprox(UnivariateQuadraticFunctionModule.trueGradient(quadraticFunction, -1), UnivariateForwardDifferenceModule.UnivariateForwardDifference(quadraticFunction, -1); atol=1e-4)
end

# ╔═╡ d90342ac-4563-4b1e-a321-e146737f10ba
md"""
## Multivariate Finite Difference
"""

# ╔═╡ 46ac28e0-5adf-11f0-2ded-0f40b02d9b11


# ╔═╡ 1b534c2b-1230-4491-a073-a1c69f924f47
begin
	# note A must be a symmetric matrix
	A = [2.0 1.0; 1.0 3.0]
	multivariateQF = MultivariateQuadraticFunctionModule.MultivariateQuadraticFunction(A)
end

# ╔═╡ 071f7bcf-ce0a-42c2-9001-c2d34562f62d
MultivariateQuadraticFunctionModule.trueGradient(multivariateQF, [1,2])

# ╔═╡ 872b6496-269a-4521-ba8e-c9ae20633e5f
MultivariateForwardDifferenceModule.MultivariateForwardDifference(multivariateQF, [3.,2.])

# ╔═╡ 145d3b24-8d54-4507-9f70-edef08b2576e
@testset "Test Multivariate Forward Difference" begin
	@test isapprox(MultivariateQuadraticFunctionModule.trueGradient(multivariateQF, [1.,2.]), MultivariateForwardDifferenceModule.MultivariateForwardDifference(multivariateQF, [1.,2.]); atol=1e-4)
	
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "71d91126b5a1fb1020e1098d9d492de2a4438fd2"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"
"""

# ╔═╡ Cell order:
# ╟─5f696645-1fad-4030-96d6-773f7f8c7201
# ╠═e7d4d847-8215-4cb2-879c-85ef9707d311
# ╟─5e36d0ac-d96f-40d3-ba22-4350f3dfd10b
# ╠═4eaa3e22-6aa5-4dfc-bd1e-da27356c0412
# ╟─083bf251-9fad-40de-89cb-4a2e67cba990
# ╠═4b26410d-00ad-4810-96dc-28c921e4fb50
# ╟─d90342ac-4563-4b1e-a321-e146737f10ba
# ╟─46ac28e0-5adf-11f0-2ded-0f40b02d9b11
# ╠═1b534c2b-1230-4491-a073-a1c69f924f47
# ╠═071f7bcf-ce0a-42c2-9001-c2d34562f62d
# ╠═872b6496-269a-4521-ba8e-c9ae20633e5f
# ╠═145d3b24-8d54-4507-9f70-edef08b2576e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
