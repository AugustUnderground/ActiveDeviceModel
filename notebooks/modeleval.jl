### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ aa866870-50f9-11eb-268e-8b614dd7f83c
using StatsBase, Plots, PlutoUI, Flux, Zygote, CUDA, BSON, PyCall, ScikitLearn, NNlib, FiniteDifferences

# ╔═╡ c7e55070-50f9-11eb-29fa-fd2d4ce6db29
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
end

# ╔═╡ 69d38de8-50fa-11eb-1e3d-4bc49627f622
md"""
# Machine Learnign Model Evaluation

## Imports
"""

# ╔═╡ da09d154-50f9-11eb-3ee8-9112221e4658
pyplot();

# ╔═╡ da08c6a6-50f9-11eb-3d36-c725e82ea052
joblib = pyimport("joblib");

# ╔═╡ a12acdfe-50ff-11eb-018f-df481ecc63df
md"""
## Loading Models
"""

# ╔═╡ e051a73a-50f9-11eb-3a7a-11906e8f508d
begin	
	modelPathᵩ = "../model/dev-2021-01-11T08:56:47.944/ptmn90";
	modelFileᵩ = modelPathᵩ * ".bson";
	trafoInFileᵩ = modelPathᵩ * ".input";
	trafoOutFileᵩ = modelPathᵩ * ".output";
	modelᵩ = BSON.load(modelFileᵩ);
	φ = modelᵩ[:model];
	trafoXᵩ = joblib.load(trafoInFileᵩ);
	trafoYᵩ = joblib.load(trafoOutFileᵩ);
	paramsXᵩ = modelᵩ[:paramsX];
	paramsYᵩ = modelᵩ[:paramsY];
end;

# ╔═╡ 1d30714c-50fa-11eb-30fc-c9b7e9ad1017
begin	
	modelPathᵨ = "../model/dev-2021-01-11T08:07:01.148/ptmn90";
	modelFileᵨ = modelPathᵨ * ".bson";
	trafoInFileᵨ = modelPathᵨ * ".input";
	trafoOutFileᵨ = modelPathᵨ * ".output";
	modelᵨ = BSON.load(modelFileᵨ);
	ρ = modelᵨ[:model];
	trafoXᵨ = joblib.load(trafoInFileᵨ);
	trafoYᵨ = joblib.load(trafoOutFileᵨ);
	paramsXᵨ = modelᵨ[:paramsX];
	paramsYᵨ = modelᵨ[:paramsY];
end;

# ╔═╡ ee6244a6-50f9-11eb-1bbf-e1757b513dfd
function predictᵩ(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoXᵩ.transform |> 
         adjoint |> φ |> adjoint |>
         trafoYᵩ.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end;

# ╔═╡ 239e7f52-50fa-11eb-1942-2b330cd2156d
function predictᵨ(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoXᵨ.transform |> 
         adjoint |> ρ |> adjoint |>
         trafoYᵨ.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end;

# ╔═╡ e9252ff0-50fc-11eb-1593-d3a6fb58186e
#begin
#	surface( vds', vgs', idᵨ'
#		   ; xaxis = "Vds [V]"
#		   , yaxis = "Vgs [V]"
#		   , zaxis = "Id [A]"
#		   , c = :blues
#		   , legend = false )
#	surface!( vgs', vds', idᵩ'; c = :reds, legend = false)
#end

# ╔═╡ f7d15572-50f9-11eb-2923-fbb666503e9d
begin
	sliderVds = @bind Vds Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	sliderVgs = @bind Vgs Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	sliderW = @bind W Slider( 7.5e-7 : 1.0e-7 : 5.0e-6
						, default = 5.0e-7, show_value = true );
	sliderL = @bind L Slider( 3e-7 : 1.0e-7 : 1.5e-6
						, default = 3e-7, show_value = true );
	
	md"""
	`Vds` = $(sliderVds) `Vgs` = $(sliderVgs)
	
	`W` = $(sliderW) `L` = $(sliderL)
	"""
end

# ╔═╡ 962c363e-541e-11eb-0880-2fb32184ca42
	Cgs, Cgd, Cds = predictᵩ( [Vds, Vds^2, Vgs, exp(Vgs), W, L] )[indexin(["cgs", "cgd", "cds"], paramsYᵩ)]

# ╔═╡ 8097b5bc-53e4-11eb-2f94-5b1c51e1db20
begin
	plt = scatter3d([Cgs], [Cgd], [Cds]
			 ; xlims = (-5e-15, 50e-15)
			 , ylims = (-5e-15, 10e-15)
			 , zlims = (-5e-15, 5e-15)
			 , xaxis = "Cgs", yaxis = "Cgd", zaxis = "Cds"
			 , title = "Capacitance", legend = false )
end

# ╔═╡ 04fa02b2-50fa-11eb-0a2b-637636057e67
begin
	vgs = collect(0.0:0.01:1.2)';
	qvgs = vgs.^2.0;
	vds = collect(0.0:0.01:1.2)';
	evds = exp.(vds);
end;

# ╔═╡ 1386b33e-50fa-11eb-0446-0b71f302857c
#begin
#	opv = reshape( Iterators.product(vgs', vds') |> collect
#		   		 , ((length(vgs) * length(vds)), 1));
#	vgd = hcat([ [o...] for o in opv ]...);
#	slen = size(vgd)[2]
#	sweepᵩ = [ vgd[2,:]'
#			 ; vgd[2,:]'.^(2.0)
#			 ; vgd[1,:]'
#			 ; exp.(vgd[1,:])'
#			 ; ones(1, slen) .* W 
#			 ; ones(1, slen) .* L ];
#	sweepᵨ = [ vgd[2,:]'
#			 ; vgd[2,:]'.^(2.0)
#			 ; vgd[1,:]'
#			 ; exp.(vgd[1,:])'
#			 ; ones(1, slen) .* W 
#			 ; ones(1, slen) .* L ];
#	predᵩ = predictᵩ(sweepᵩ);
#	predᵨ = predictᵨ(sweepᵨ);
#	idᵩ = reshape( predᵩ[first(indexin(["id"], paramsYᵩ)), :]
#				 , (Int(sqrt(slen)), Int(sqrt(slen))));
#	idᵨ = reshape( predᵨ[first(indexin(["id"], paramsYᵨ)), :]
#				 , (Int(sqrt(slen)), Int(sqrt(slen))));
#end;

# ╔═╡ Cell order:
# ╟─69d38de8-50fa-11eb-1e3d-4bc49627f622
# ╠═aa866870-50f9-11eb-268e-8b614dd7f83c
# ╠═c7e55070-50f9-11eb-29fa-fd2d4ce6db29
# ╠═da09d154-50f9-11eb-3ee8-9112221e4658
# ╠═da08c6a6-50f9-11eb-3d36-c725e82ea052
# ╟─a12acdfe-50ff-11eb-018f-df481ecc63df
# ╠═e051a73a-50f9-11eb-3a7a-11906e8f508d
# ╠═1d30714c-50fa-11eb-30fc-c9b7e9ad1017
# ╠═ee6244a6-50f9-11eb-1bbf-e1757b513dfd
# ╠═239e7f52-50fa-11eb-1942-2b330cd2156d
# ╠═e9252ff0-50fc-11eb-1593-d3a6fb58186e
# ╟─f7d15572-50f9-11eb-2923-fbb666503e9d
# ╠═8097b5bc-53e4-11eb-2f94-5b1c51e1db20
# ╟─962c363e-541e-11eb-0880-2fb32184ca42
# ╠═04fa02b2-50fa-11eb-0a2b-637636057e67
# ╠═1386b33e-50fa-11eb-0446-0b71f302857c
