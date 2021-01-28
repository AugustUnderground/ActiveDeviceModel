### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# ╔═╡ 980c5ef6-614e-11eb-394a-65db3f0c3156
using OnlineStats, DataFrames, JLD2, Plots, Random, StatsBase

# ╔═╡ bd417ce2-614e-11eb-1576-df8b077f3128
data = jldopen("../../data/ptmn90.jld") do file
	file["database"];
end;

# ╔═╡ 198399d4-6151-11eb-050d-bd04500d4829
begin
	numSamples = 666666;
	rngSeed = 666;
end

# ╔═╡ 166685d6-6151-11eb-3e7c-5798ae2582ea
sample = data[ StatsBase.sample( MersenneTwister(rngSeed)
                               , 1:(data |> size |> first)
                               , pweights(data.id)
                               , numSamples
                               ; replace = false )
			 , : ];

# ╔═╡ cf1f8398-614e-11eb-13d4-89f3c4be575d
plot( plot(fit!(ExpandingHist(1000), data.id))
	, plot(fit!(ExpandingHist(1000), sample.id)) )

# ╔═╡ eab691b8-6150-11eb-0f34-3990164730c8
plot( plot(fit!(OrderStats(1000), data.id))
	, plot(fit!(OrderStats(1000), sample.id)) )

# ╔═╡ Cell order:
# ╠═980c5ef6-614e-11eb-394a-65db3f0c3156
# ╠═bd417ce2-614e-11eb-1576-df8b077f3128
# ╠═198399d4-6151-11eb-050d-bd04500d4829
# ╠═166685d6-6151-11eb-3e7c-5798ae2582ea
# ╠═cf1f8398-614e-11eb-13d4-89f3c4be575d
# ╠═eab691b8-6150-11eb-0f34-3990164730c8
