### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 377ae5da-1032-42d1-8010-5bd1887fde7a
begin
	include("../../utils/GradientEstimators/AutomaticDifferentiation.jl")
	include("../../utils/TestFunctions.jl")
	using Test
	using .UnivariateQuadraticFunctionModule
	using .UnivariateForwardAutoDiffModule
end

# ╔═╡ 53712054-8545-488c-a6c1-0bbbcbc5c979
using.MultivariateQuadraticFunctionModule

# ╔═╡ 76891c20-5f68-11f0-0dc5-af2d69d24278
md"""
# Forward Automatic Differentiation Testing
## Univariate Forward Automatic Differentiation
import functions
"""

# ╔═╡ fa7e0f99-ab4d-463b-b5d5-e861528c54bb
begin
f(x) = x * x + 3.0 * x

x = D(2.0, 1.0)  # Value = 2.0, Derivative = 1.0
y = f(x)
println("Value: ", y.f[1])
println("Derivative: ", y.f[2])
end


# ╔═╡ c604a5a7-d5bb-4bff-a540-02e5867259bb
md"""
## Multivariate Forward Automatic Differentiation
import functions
"""

# ╔═╡ 927708dd-e592-4bde-9f68-baaf9b2af838


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
# ╟─76891c20-5f68-11f0-0dc5-af2d69d24278
# ╠═377ae5da-1032-42d1-8010-5bd1887fde7a
# ╠═fa7e0f99-ab4d-463b-b5d5-e861528c54bb
# ╟─c604a5a7-d5bb-4bff-a540-02e5867259bb
# ╠═53712054-8545-488c-a6c1-0bbbcbc5c979
# ╠═927708dd-e592-4bde-9f68-baaf9b2af838
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
