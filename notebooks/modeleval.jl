### A Pluto.jl notebook ###
# v0.12.19

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
using Plots, PlutoUI, Flux, Zygote, CUDA, BSON, PyCall, ScikitLearn, NNlib, FiniteDifferences

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
#plotly();

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
	modelPathᵧ = "../model/des-2021-01-12T13:58:59.001/ptmn90";
	modelFileᵧ = modelPathᵧ * ".bson";
	trafoInFileᵧ = modelPathᵧ * ".input";
	trafoOutFileᵧ = modelPathᵧ * ".output";
	modelᵧ = BSON.load(modelFileᵧ);
	ρ = modelᵧ[:model];
	trafoXᵧ = joblib.load(trafoInFileᵧ);
	trafoYᵧ = joblib.load(trafoOutFileᵧ);
	paramsXᵧ = modelᵧ[:paramsX];
	paramsYᵧ = modelᵧ[:paramsY];
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
function predictᵧ(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoXᵧ.transform |> 
         adjoint |> ρ |> adjoint |>
         trafoYᵧ.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end;

# ╔═╡ 6a642ecc-559e-11eb-17ba-e73320d0ba0b
begin	
	modelPathₙ = "../model/ptmn90-2021-01-13T11:44:32.437/ptmn90";
	modelFileₙ = modelPathₙ * ".bson";
	trafoInFileₙ = modelPathₙ * ".input";
	trafoOutFileₙ = modelPathₙ * ".output";
	modelₙ = BSON.load(modelFileₙ);
	φₙ = modelₙ[:model];
	trafoXₙ = joblib.load(trafoInFileₙ);
	trafoYₙ = joblib.load(trafoOutFileₙ);
	paramsXₙ = modelₙ[:paramsX];
	paramsYₙ = modelₙ[:paramsY];
	
	function predictₙ(X)
 		rY = ((length(size(X)) < 2) ? [X'] : X') |>
        	 trafoXₙ.transform |> 
         	 adjoint |> φₙ |> adjoint |>
         	 trafoYₙ.inverse_transform |> 
         	 adjoint
  		return Float64.(rY)
	end;
end;

# ╔═╡ b9707462-559e-11eb-083e-7952cd88a191
begin	
	modelPathₚ = "../model/ptmp90-2021-01-13T12:04:05.819/ptmp90";
	modelFileₚ = modelPathₚ * ".bson";
	trafoInFileₚ = modelPathₚ * ".input";
	trafoOutFileₚ = modelPathₚ * ".output";
	modelₚ = BSON.load(modelFileₚ);
	φₚ = modelₚ[:model];
	trafoXₚ = joblib.load(trafoInFileₚ);
	trafoYₚ = joblib.load(trafoOutFileₚ);
	paramsXₚ = modelₚ[:paramsX];
	paramsYₚ = modelₚ[:paramsY];
	
	function predictₚ(X)
 		rY = ((length(size(X)) < 2) ? [X'] : X') |>
        	 trafoXₚ.transform |> 
         	 adjoint |> φₚ |> adjoint |>
         	 trafoYₚ.inverse_transform |> 
         	 adjoint
  		return Float64.(rY)
	end;
end;

# ╔═╡ e0a0602e-559e-11eb-1d39-55a55a554d88
function nmos(Vgs, Vds, W, L)
	xₙ = [Vds ; Vds.^2 ; Vgs ; exp.(Vgs); W ; L];
	yₙ = predictₙ(xₙ);
	return yₙ
end;

# ╔═╡ 7f717e72-559f-11eb-01a1-37babe71157b
function pmos(Vgs, Vds, W, L)
	xₚ = [Vds ; Vds.^2 ; Vgs ; exp.(Vgs); W ; L];
	yₚ = predictₙ(xₚ);
	return yₚ
end;

# ╔═╡ 7d0501a4-54c3-11eb-3e5c-275701e033ea
md"""
## Exploring the Device Model
"""

# ╔═╡ 04fa02b2-50fa-11eb-0a2b-637636057e67
begin
	vgs = collect(0.0:0.01:1.2)';
	qvgs = vgs.^2.0;
	vds = collect(0.0:0.01:1.2)';
	evds = exp.(vds);
end;

# ╔═╡ e9252ff0-50fc-11eb-1593-d3a6fb58186e
#begin
# 	surface( vds', vgs',idᵩ'
#		   ; xaxis = "Vds [V]"
#		   , yaxis = "Vgs [V]"
#		   , zaxis = "Id [A]"
#		   , c = :blues
#		   , legend = false )
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

# ╔═╡ 1386b33e-50fa-11eb-0446-0b71f302857c
begin
	opv = reshape( Iterators.product(vgs', vds') |> collect
		   		 , ((length(vgs) * length(vds)), 1));
	vgd = hcat([ [o...] for o in opv ]...);
	slen = size(vgd)[2]
	sweepᵩ = [ vgd[2,:]'
			 ; vgd[2,:]'.^(2.0)
			 ; vgd[1,:]'
			 ; exp.(vgd[1,:])'
			 ; ones(1, slen) .* W 
			 ; ones(1, slen) .* L ];
	predᵩ = predictₙ(sweepᵩ);
	idᵩ = reshape( predᵩ[first(indexin(["id"], paramsYₙ)), :]
				 , (Int(sqrt(slen)), Int(sqrt(slen))));
end;

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

# ╔═╡ 46da2f52-5b1c-11eb-3476-73665e0f2366
md"""
### Partial Derivatives
"""

# ╔═╡ 5291467a-5b1d-11eb-1904-0344a8ac78cf
begin
	vLen = 121;
	#pV = range(0.0, length = vLen, stop = 1.2) |> collect;
	#sV = [repeat(pV, inner=[size(pV, 1)]) repeat(pV, outer=[size(pV,1)])]
	#pVgs, pVds = eachcol(sV);
	#pLen = first(size(sV));
	
	pVgs = fill(0.6, vLen);
	pVds = range(0.0, length = vLen, stop = 1.2) |> collect;
	
	pW = fill(2.0e-6, vLen);
	pL = fill(3.0e-7, vLen);
	
	pX = [ pVgs |> collect |> adjoint
		 ; pVds |> collect |> adjoint
		 ; pW |> collect |> adjoint
		 ; pL |> collect |> adjoint ];
end;

# ╔═╡ 8e79cbc0-5bdc-11eb-06a5-29dcd6b17399
function drainCurrent(vgs, vds, w, l)
	pY = predictₙ( [ vgs'  ; (vgs.^2)'
				   ; vds' ; (ℯ.^vds)'
				   ; w' ; l' ] );
	return pY[first(indexin(["id"], paramsYₙ)), :][1];
end;

# ╔═╡ bed655ac-5bdb-11eb-34db-53f5a73d8f1b
pId = [ drainCurrent(x...) for x in eachcol(pX) ];

# ╔═╡ d0eae564-5bc0-11eb-1684-b9973a048479
begin
	plot(pVds, pId)
	#surface!(pVgs, pVds, ∂id_∂LGrid; c = :blues, legend = false)
	#surface!(pVgs, pVds, ∂id_∂WGrid; c = :greens, legend = false)
end

# ╔═╡ 944ca788-5bbf-11eb-100e-bb1ed0c52faf
#begin
	#∂id = hcat([ grad(central_fdm(5,1), id, px)[1] for px in eachcol(pX) ]...);
	#∂id_∂Vgs, ∂id_∂Vgs, ∂id_∂W, ∂id_∂L = eachrow(∂id);
	#∂id_∂LGrid = reshape(∂id_∂L, (Int(sqrt(pLen)), Int(sqrt(pLen))));
	#∂id_∂WGrid = reshape(∂id_∂W, (Int(sqrt(pLen)), Int(sqrt(pLen))));
#end

# ╔═╡ 843a61ec-54e3-11eb-0029-fbfd2976f29b
md"""
## Exploring Design Model

$$\begin{bmatrix}
L \\ i_{\text{d}} \\ \frac{g_{\text{m}}}{i_{\text{d}}} \\ V_{\text{ds}} \\
\end{bmatrix}
\mapsto
\begin{bmatrix}
V_{\text{gs}} \\ \frac{i_{\text{d}}}{W} \\ A_{0} \\ f_{\text{ug}} \\
\end{bmatrix}$$
"""

# ╔═╡ 0e78eab8-54e4-11eb-07e5-0f5b4e63d5e8
begin
	sliderVdsᵧ = @bind Vdsᵧ Slider( 0.01 : 0.01 : 1.20
								  , default = 0.6, show_value = true );
	sliderIdᵧ = @bind Idᵧ Slider( 10e-6 : 1.0e-7 : 75e-6
						        , default = 25e-6, show_value = true );
	sliderLᵧ = @bind Lᵧ Slider( 3e-7 : 1.0e-7 : 1.5e-6
							  , default = 3e-7, show_value = true );
	sliderGᵧ = @bind gmidᵧ Slider( 1 : 0.5 : 25
							  , default = 10, show_value = true );
	md"""
	`L` = $(sliderLᵧ) m
	
	`Id` = $(sliderIdᵧ) A
	
	`gm/id` = $(sliderGᵧ) S/A
	
	`Vds` =$(sliderVdsᵧ) V
	"""
end

# ╔═╡ 8cb6ff56-54e3-11eb-0c02-713c6d2be8b5
begin
	gmid = 1.0:0.25:25.0;
	len  = length(gmid);
	
	xᵧ = [ fill(Lᵧ, 1, len)
		 ; fill(Idᵧ, 1, len)
		 ; collect(gmid)'
		 ; (gmid .* Idᵧ)'
		 ; fill(Vdsᵧ, 1, len)
		 ; exp.(fill(Vdsᵧ, 1, len)) ];
	pᵧ = predictᵧ(xᵧ);	
end

# ╔═╡ 0b4ab9d6-5661-11eb-1c04-21ea0f3916e1
begin
	idWplot = plot( gmid, pᵧ[2,:]
				  ; yscale = :log10
				  , yaxis = "id/W [A]"
				  , xaxis = "gm/id [S/A]"
				  , title = "Current Density over gm/id"
				  , legend = false );
	A0plot = plot( gmid, pᵧ[3,:]
				  ; yscale = :log10
				  , yaxis = "A0"
				  , xaxis = "gm/id [S/A]"
				  , title = "Self Gain over gm/id"
				  , legend = false );
	plot( idWplot, A0plot, layout = (1,2));
end

# ╔═╡ 4702177c-5658-11eb-1912-6b7b95c8f221
#begin
#	∇gmid = (x) -> grad(central_fdm(5,1), (x) -> predictᵧ(x)[2,1], x) |> first;
#	∂gmid = hcat(map(∇gmid, eachcol(xᵧ))...)[1,:];
#	∇gmid(xᵧ[:,1])
#end

# ╔═╡ 3dbf3974-54fa-11eb-1901-c310b0dd8685
md"""
### Optimization

$(PlutoUI.LocalResource("./symamp.png"))

#### Find Operating Point

1. Define KCL-Error-Function
1. Must contain all Transistors
1. Add Currents add nodes
1. Still done manually

Known Volatges:

$$\phi_{dd} = 1.2\,\text{V}$$
$$\phi_{ss} = 0.0\,\text{V}$$
$$\phi_{I1} = 0.7\,\text{V}$$
$$\phi_{I2} = 0.7\,\text{V}$$
$$\phi_{bias} = 0.8\,\text{V}$$

Find:

$$\phi_{0} = \begin{bmatrix}
\phi_{x} \\
\phi_{u} \\
\phi_{w} \\
\phi_{c} \\
\phi_{b} \\
\phi_{o} \\
\end{bmatrix}$$

So that KCL checks out.

#### Find W/L for all transistors

1. Based on **Find Operating Point**
1. Define Cost-Function with MNA
1. Optimize W/L

"""

# ╔═╡ b6616c00-55a0-11eb-346b-99831a762e03
begin
	ϕDD = 1.2;
	ϕSS = 0.0;
	ϕI1 = 0.7;
	ϕI2 = 0.7;

	ib = 50e-7;
	
	W₁₂ = 2e-6;
	W₃₄ = 4e-6;
	W₅₆ = 4e-6;
	W₇₈ = 3e-6;
	W₉₀ = 3e-6;
	
	Lₘᵢₙ = 3e-7;
end;

# ╔═╡ 529afb34-55a0-11eb-36e7-45fdb7453178
function kcl(ϕ)
    #ϕx, ϕu, ϕw, ϕc, ϕo, ϕB = ϕ;
    id1 , id2 , id3 , id4 , id5 , id6 , id7 , id8 , id9 , id0 = first.(
        [ nmos((ϕI1 - ϕ[1]), (ϕ[2] - ϕ[1]), W₁₂, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos((ϕI2 - ϕ[1]), (ϕ[3] - ϕ[1]), W₁₂, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , pmos((ϕDD - ϕ[2]), (ϕDD - ϕ[2]), W₃₄, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , pmos((ϕDD - ϕ[2]), (ϕDD - ϕ[4]), W₃₄, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , pmos((ϕDD - ϕ[3]), (ϕDD - ϕ[3]), W₅₆, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , pmos((ϕDD - ϕ[3]), (ϕDD - ϕ[5]), W₅₆, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , nmos(ϕ[6], ϕ[6], W₇₈, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos(ϕ[6], ϕ[1], W₇₈, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos(ϕ[4], ϕ[4], W₉₀, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos(ϕ[4], ϕ[5], W₉₀, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1] ]);
    ΔϕDD = abs((id4 + id3) - (id5 + id6));
    Δϕu  = abs(id3 - id1);
    Δϕw  = abs(id5 - id2);
    Δϕx  = abs((id1 - id2) - id8);
    Δϕc  = abs(id4 - id9);
    ΔϕB  = abs(id7 - ib);
    Δϕo  = abs(id5 - id0);
    return [ ΔϕDD, Δϕu , Δϕw , Δϕx , Δϕc , ΔϕB , Δϕo ];
end;

# ╔═╡ 29ab0e1e-559e-11eb-2d50-cbfd0e603acb
function cost()
	return (1 / abs(sum(kcl(x))))
end;

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
# ╠═6a642ecc-559e-11eb-17ba-e73320d0ba0b
# ╠═b9707462-559e-11eb-083e-7952cd88a191
# ╠═e0a0602e-559e-11eb-1d39-55a55a554d88
# ╠═7f717e72-559f-11eb-01a1-37babe71157b
# ╟─7d0501a4-54c3-11eb-3e5c-275701e033ea
# ╠═04fa02b2-50fa-11eb-0a2b-637636057e67
# ╠═1386b33e-50fa-11eb-0446-0b71f302857c
# ╠═e9252ff0-50fc-11eb-1593-d3a6fb58186e
# ╟─8097b5bc-53e4-11eb-2f94-5b1c51e1db20
# ╟─962c363e-541e-11eb-0880-2fb32184ca42
# ╟─f7d15572-50f9-11eb-2923-fbb666503e9d
# ╟─46da2f52-5b1c-11eb-3476-73665e0f2366
# ╠═d0eae564-5bc0-11eb-1684-b9973a048479
# ╠═5291467a-5b1d-11eb-1904-0344a8ac78cf
# ╠═8e79cbc0-5bdc-11eb-06a5-29dcd6b17399
# ╠═bed655ac-5bdb-11eb-34db-53f5a73d8f1b
# ╠═944ca788-5bbf-11eb-100e-bb1ed0c52faf
# ╟─843a61ec-54e3-11eb-0029-fbfd2976f29b
# ╟─0b4ab9d6-5661-11eb-1c04-21ea0f3916e1
# ╟─0e78eab8-54e4-11eb-07e5-0f5b4e63d5e8
# ╟─8cb6ff56-54e3-11eb-0c02-713c6d2be8b5
# ╠═4702177c-5658-11eb-1912-6b7b95c8f221
# ╟─3dbf3974-54fa-11eb-1901-c310b0dd8685
# ╠═b6616c00-55a0-11eb-346b-99831a762e03
# ╠═529afb34-55a0-11eb-36e7-45fdb7453178
# ╠═29ab0e1e-559e-11eb-2d50-cbfd0e603acb
