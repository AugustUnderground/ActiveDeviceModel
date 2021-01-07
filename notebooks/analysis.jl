### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 4222971e-50cb-11eb-10f6-759d08092c63
using StatsPlots, Printf, StatsBase, Statistics, DataFrames, JLD2, MLDataUtils

# ╔═╡ 189f8082-50cb-11eb-0ea6-61cc92c9a3f9
md"""
# Data Analysis

Investigate correlations of features in OPP Simulation Data.

## Load Libraries
"""

# ╔═╡ f66deaac-50cb-11eb-36f6-eb565ab7be7b
md"""
## Load the Data Frame
"""

# ╔═╡ 6544217c-50cb-11eb-1a8b-b1dc57e82d0f
dataFramePath = "../data/ptmn90.jld";

# ╔═╡ 755e1504-50cb-11eb-2966-33e9bd326e5f
dataFrame = jldopen(dataFramePath, "r") do file
	file["database"];
end;

# ╔═╡ d4a551ce-50cd-11eb-2a13-a9302fc162db
md"""
### General Description
"""

# ╔═╡ 99792352-50cb-11eb-0a11-e5a7f13136ca
describe(dataFrame)

# ╔═╡ a89d43ac-50e1-11eb-3a6a-9fe8e6d248e8
begin
	numSamples = 5000;
	idxSamples = rand(1:(dataFrame |> size |> first), numSamples);
end;

# ╔═╡ 8e4748a8-50e2-11eb-14e3-d55fcfdf2ccc
sampleData = dataFrame[idxSamples, :];

# ╔═╡ 71ad483e-50ef-11eb-0bca-8bd53995e25a
@df sampleData corrplot(cols([8 9 7 12 11]))

# ╔═╡ Cell order:
# ╟─189f8082-50cb-11eb-0ea6-61cc92c9a3f9
# ╠═4222971e-50cb-11eb-10f6-759d08092c63
# ╟─f66deaac-50cb-11eb-36f6-eb565ab7be7b
# ╠═6544217c-50cb-11eb-1a8b-b1dc57e82d0f
# ╠═755e1504-50cb-11eb-2966-33e9bd326e5f
# ╟─d4a551ce-50cd-11eb-2a13-a9302fc162db
# ╠═99792352-50cb-11eb-0a11-e5a7f13136ca
# ╠═a89d43ac-50e1-11eb-3a6a-9fe8e6d248e8
# ╠═8e4748a8-50e2-11eb-14e3-d55fcfdf2ccc
# ╠═71ad483e-50ef-11eb-0bca-8bd53995e25a
