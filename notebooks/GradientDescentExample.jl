### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 0741def0-61f3-11f0-1d77-39ade2248515
begin
	include("../utils/GradientEstimators/GradientEstimatorType.jl")
	include("../utils/GradientEstimators/ForwardDifference.jl")
	include("../utils/TestFunctions.jl")
	include("../algorithims/GradientDescent.jl")
	using Test
	using .UnivariateQuadraticFunctionModule, .MultivariateQuadraticFunctionModule
	using .UnivariateForwardDifferenceModule
	using .GradientDescentModule
end

# ╔═╡ 7496a50f-c53b-42ab-b13d-7e7cf317203f
qf = UnivariateQuadraticFunction(1,2,3)

# ╔═╡ 1b79e529-665a-4c1a-bc8d-fdc0d35f87ff
gradEstimator = UnivariateForwardDifference(1e-5)

# ╔═╡ a299d90f-a990-4af7-af37-b2f18322bcf9
qf(1)

# ╔═╡ 39ff5d36-680b-4a27-bd61-89f309a6ce02
x0 = 1.

# ╔═╡ aa660efb-887c-46b5-9b04-0ad6d83b9445
# a_0 = qf(x0)

# ╔═╡ 1c6265c9-776b-467b-b780-0c84d0dee454
tol = 1e-3

# ╔═╡ b765c651-35ef-4dac-9fd9-c121f78515a0
println(typeof(gradEstimator))

# ╔═╡ 7775503a-28f4-40ac-9820-5dcd232f2d97
println("Is GradientEstimator? ", isa(gradEstimator, GradientEstimator))

# ╔═╡ bad81261-becd-448c-a024-9faf52e16150
# gradient(qf, 1)
grad_func = gradient(gradEstimator, qf, 1)

# ╔═╡ afa9b8bf-8252-41c1-90be-40c2706069c0
println(typeof(grad_func))

# ╔═╡ 214673b0-6e28-4fda-b577-79cedcc46db1
# ╠═╡ disabled = true
#=╠═╡
# a_1 = a_0 - 0.01 * UnivariateForwardDifference(qf, a_0[1])
  ╠═╡ =#

# ╔═╡ 2c994603-f4bd-4abe-9594-56b8a4b0f4e7
# begin
	# println(typeof(a_0))
	# println(typeof(a_1))
# end

# ╔═╡ 444061d7-21f3-4079-80fe-1297b2cc0f1c
begin
	
    step_size = 0.01
    a_0 = [x0]  # assuming x0 is your initial point
    a_1 = a_0 - step_size * gradient(gradEstimator, qf, a_0[1])
	result = gradient(gradEstimator, qf, a_0[1])
	println(typeof(result))
	println(result)
end

# ╔═╡ ff9eb1d4-d077-42ac-ab88-d9de97336ea0
# begin
# 	a_0 = qf(x0)
# 	a_1 = a_0 - 0.01 * UnivariateForwardDifference(qf, a_0[1])
# 	while abs(a_1[1] - a_0[1]) > tol
# 		a_0 = a_1
# 		a_1 = a_0[1] - step_size * UnivariateForwardDifference(qf, a_0[1])
# 	end
# end

begin
    while abs(a_1[1] - a_0[1]) > tol
        a_0 = a_1
        a_1 = a_0[1] - step_size * gradient(gradEstimator, qf, a_0[1])[1]
    end
end

# ╔═╡ 5ecf7825-6137-44de-a00c-f3a7cfe83991
# begin
# 	include("../algorithims/GradientDescent.jl")
# 	include("../utils/GradientEstimators/GradientEstimatorType.jl")
# 	using .GradientDescentModule
# end

# ╔═╡ 632ec057-d8bd-4c59-8e39-c1715eef21e4
println("Is GradientEstimator? ", isa(gradEstimator, GradientEstimator))

# ╔═╡ b32bcf09-4386-4877-9462-7a36aec5d36d
GradientDescent(qf, 1.0, gradEstimator)

# ╔═╡ b3835283-219d-4b46-a90a-2a4e4f96c3af
A = [2.0 3.0; 3.0 5.0]


# ╔═╡ d3b16900-2be2-4cc9-82db-50738252b22f
qfm = MultivariateQuadraticFunction(A)

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
# ╠═0741def0-61f3-11f0-1d77-39ade2248515
# ╠═7496a50f-c53b-42ab-b13d-7e7cf317203f
# ╠═1b79e529-665a-4c1a-bc8d-fdc0d35f87ff
# ╠═a299d90f-a990-4af7-af37-b2f18322bcf9
# ╠═39ff5d36-680b-4a27-bd61-89f309a6ce02
# ╠═aa660efb-887c-46b5-9b04-0ad6d83b9445
# ╠═1c6265c9-776b-467b-b780-0c84d0dee454
# ╠═b765c651-35ef-4dac-9fd9-c121f78515a0
# ╠═7775503a-28f4-40ac-9820-5dcd232f2d97
# ╠═bad81261-becd-448c-a024-9faf52e16150
# ╠═afa9b8bf-8252-41c1-90be-40c2706069c0
# ╠═214673b0-6e28-4fda-b577-79cedcc46db1
# ╠═2c994603-f4bd-4abe-9594-56b8a4b0f4e7
# ╠═444061d7-21f3-4079-80fe-1297b2cc0f1c
# ╠═ff9eb1d4-d077-42ac-ab88-d9de97336ea0
# ╠═5ecf7825-6137-44de-a00c-f3a7cfe83991
# ╠═632ec057-d8bd-4c59-8e39-c1715eef21e4
# ╠═b32bcf09-4386-4877-9462-7a36aec5d36d
# ╠═b3835283-219d-4b46-a90a-2a4e4f96c3af
# ╠═d3b16900-2be2-4cc9-82db-50738252b22f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
