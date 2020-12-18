### A Pluto.jl notebook ###
# v0.12.17

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

# ╔═╡ 647ab7f4-3e12-11eb-29a9-cd98784528f1
using Flux, Lazy, Printf, Plots, StatsBase, PyCall, BSON, Zygote, CUDA, mna, PlutoUI, Optim, LinearAlgebra, Calculus

# ╔═╡ 3c9843ae-3fc1-11eb-2da4-6314f7556996
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
end

# ╔═╡ 3d9fe546-3e12-11eb-3e0d-7f5e9d423e92
md"""
# Analog Building Blocks
"""

# ╔═╡ 39e96b2e-3fc6-11eb-339c-3dbc92012d34
plotly();

# ╔═╡ 8d5934b6-3fc0-11eb-3770-97600d88d634
joblib = pyimport("joblib");

# ╔═╡ 80f4d512-3fbe-11eb-1548-01c553692bf4
md"""
### Primitive Devices
		
A _MOSFET_ is defined as a `struct` with access to the NN Model and data Transformations.
"""

# ╔═╡ 9635c21a-4135-11eb-2b1d-65e5ce383bce
struct MOSFET
	type
	model
	paramsX
	paramsY
	trafoX
	trafoY
end

# ╔═╡ 2c96d8d6-3fc0-11eb-1835-450c15fec5ac
devicePath = "./model/dev-2020-12-18T13:26:12.707/ptmn90";

# ╔═╡ 84508ff2-3fc2-11eb-2ad7-77c2953a0515
function predict(μ, X)
	rY = ((length(size(X)) < 2) ? [X'] : X') |>
    	 μ.trafoX.transform |> 
         adjoint |> μ.model |> adjoint |>
         μ.trafoY.inverse_transform |> 
         adjoint
	return Float64.(rY)
end;

# ╔═╡ 0669423a-4136-11eb-35dd-d9e990ecb321
begin
	model = BSON.load(devicePath * ".bson");
	nmos = MOSFET( "n"
				 , model[:model]
				 , model[:paramsX]
				 , model[:paramsY]
				 , joblib.load(devicePath * ".input")
				 , joblib.load(devicePath * ".output") );
end;

# ╔═╡ f9db2f6c-4115-11eb-1a7c-9bbb7f651ec5
opp = (mos, prd, param) -> prd[first(indexin([param], mos.paramsY)), :];

# ╔═╡ 0cbd8a4e-410c-11eb-07ef-1f2cce6153ed
begin
	Lmin = 3e-7;
	Lmax = 3e-6;
	Wmin = 1e-6;
	Wmax = 5e-6;
	Vgsmin = 0.0;
	Vgsmax = 1.2;
	Vgsnom = 0.6;
	Vdsmin = 0.0;
	Vdsmax = 1.2;
	Vdsnom = 0.6;
	sweepLen = 121;
	
md"""
### Design Limitations
	
$$L_{\text{min}} = 300\,\text{nm}$$
$$L_{\text{max}} = 3\,\mu\text{m}$$

$$W_{\text{min}} = 1\,\mu\text{m}$$
$$W_{\text{max}} = 5\,\mu\text{m}$$

$$V_{\text{gs,min}} = 0.0\,\text{V}$$
$$V_{\text{gs,max}} = 1.2\,\text{V}$$
	
$$V_{\text{ds,min}} = 0.0\,\text{V}$$
$$V_{\text{ds,max}} = 1.2\,\text{V}$$
"""
end

# ╔═╡ ed52203c-410d-11eb-2f54-e7ee56faa0de
md"""
Some **ranges** and **constants** for plotting:
"""

# ╔═╡ 9de1eb8e-4105-11eb-39d3-c7d85be8f5ab
begin
	rangeVgs = range(Vgsmin, stop = Vgsmax, length = sweepLen);
	rangeQVgs = rangeVgs.^2.0;
	rangeVds = range(Vdsmin, stop = Vdsmax, length = sweepLen);
	rangeQVds = rangeVds.^2.0;
	rangeVdgs = rangeVds .* rangeVgs;
	rangeW = range(Wmin, step = 0.5e-7, length = sweepLen);
	rangeL = range(Lmin, stop = Lmax, length = sweepLen);
end;

# ╔═╡ bcdbf078-3fc2-11eb-19f9-bbddbdbca559
md"""
### Testing the Device Model
"""

# ╔═╡ 76cd2dfa-4115-11eb-1c8b-095081973632
#plot( plot( rangeVgs, idₜ
#		  , xaxis = "Vgs", yaxis = "Id"
#		  , title = "Transfer Characteristic" )
#	, plot( rangeVds, idₒ
#		  , xaxis = "Vds", yaxis = "Id"
#		  , title = "Output Characterisitc" )
#	, layout = (1, 2), legend = false)

# ╔═╡ 6301e07c-4138-11eb-1606-2d9b676f2bdd
md"""
### Testing the Design Model

Defined by two functions:

$$f_{\gamma} ( \frac{g_{\text{m}}}{i_{\text{d}}} , V_{\text{ds}}, L ) \mapsto
[ A_{0}, f_{\text{ug}}, \frac{i_{\text{d}}}{W}, V_{\text{gs}} ]$$ 

$$f_{\nu} ( v_{\text{dsat}} , V_{\text{ds}}, L ) \mapsto
[ A_{0}, f_{\text{ug}}, \frac{i_{\text{d}}}{W}, V_{\text{gs}} ]$$ 
"""

# ╔═╡ 4aff32a0-4149-11eb-2efd-a34ab9714a4e
struct DesignFunction
	model
	paramsX
	paramsY
	trafoX
	trafoY
end;

# ╔═╡ f5df8218-413b-11eb-385d-6dfd7ca750f8
begin
	gmidPath = "./model/des-2020-12-18T13:42:19.252/ptmn90";
	gmidFile = BSON.load(gmidPath * ".bson");
	γ = DesignFunction( gmidFile[:model]
	                  , ["L", "gmid", "Vds", "QVds"]
	                  , [ "A0", "A0Log", "fug", "fugLog"
                        , "idW", "idWLog", "Vgs", "QVgs" ]
	                  , joblib.load(gmidPath * ".input")
                      , joblib.load(gmidPath * ".output") );
	
	vdsatPath = "./model/des-2020-12-18T13:48:31.766/ptmn90";
	vdsatFile = BSON.load(vdsatPath * ".bson");
	ν = DesignFunction( vdsatFile[:model]
	                  , ["L", "vdsat", "Vds", "QVds"]
	                  , [ "A0", "A0Log", "fug", "fugLog"
                        , "idW", "idWLog", "Vgs", "QVgs" ]
	                  , joblib.load(vdsatPath * ".input")
                      , joblib.load(vdsatPath * ".output") );
end;

# ╔═╡ d54b9bee-413f-11eb-3ea1-bff788c78330
begin
	fᵧ = (gmid, vds, l) -> predict(γ, [l; gmid; vds; (vds.^2)])[[1,3,5,7],:];
	fᵥ = (vdsat, vds, l) -> predict(ν, [l; vdsat; vds; (vds.^2)])[[1,3,5,7],:];
end;

# ╔═╡ 693f007c-4107-11eb-194f-25b6811ceee2
begin
	slVds = @bind Vds Slider(rangeVds, default = Vdsnom, show_value = true);
	slVgs = @bind Vgs Slider(rangeVgs, default = Vgsnom, show_value = true);
	slW = @bind W Slider(rangeW, default = Wmin, show_value = true);
	slL = @bind L Slider(rangeL, default = Lmin, show_value = true);
	
	md"""
	`Vds` = $(slVds) `Vgs` = $(slVds)
	
	`W` = $(slW) `L` = $(slL)
	"""
end

# ╔═╡ 658008b2-4109-11eb-2128-951666af5ceb
begin
	constW = ones(1,sweepLen) .* W;
	constL = ones(1,sweepLen) .* L;
	constVgs = ones(1,sweepLen) .* Vgs;
	constQVgs = constVgs.^2.0;
	constVds = ones(1,sweepLen) .* Vds;
	constQVds = constVds.^2.0;
	constVdgs = constVds .* constVgs;
	constVbs = zeros(1,sweepLen);
end;

# ╔═╡ de21a462-3fc2-11eb-3121-db02f67519cc
begin
	Xₜ = [ rangeVgs'
		 ; constVds
		 ; constVbs
		 ; constW
		 ; constL
		 ; rangeQVgs'
		 ; constQVds
		 ; constVds .* rangeVgs' ];
	pₜ = predict(nmos, Xₜ);
	idₜ = pₜ[first(indexin(["id"], nmos.paramsY)), :];
end;

# ╔═╡ f10ee5e6-3fc2-11eb-2916-3578e545b0bf
begin
	Xₒ = [ constVgs
		 ; rangeVds'
		 ; constVbs
		 ; constW
		 ; constL
		 ; constQVgs
		 ; rangeQVds'
		 ; rangeVds' .* constVgs ];
	pₒ = predict(nmos, Xₒ);
	idₒ = pₒ[first(indexin(["id"], nmos.paramsY)), :];
end;

# ╔═╡ 412d1876-4146-11eb-1dcc-31217a410d35
begin
	gmid = collect(1:0.1:25)';
	vds = ones(1,length(gmid)) .* Vds;
	l = ones(1,length(gmid)) .* L;
	idW = (gmid,vds,l) -> fᵧ(gmid, vds, l)[3,:];
end;

# ╔═╡ d84ad25a-414f-11eb-2d7d-eb31e96d3af4
plot( gmid', idW(gmid, ones(1,length(gmid)) .* Vds, ones(1, length(gmid)) .* L)
	; xaxis = "gm/id", yaxis = "id/W", yscale = :log10, legend = nothing)

# ╔═╡ c850e59c-4154-11eb-1931-cb3c7e18356d
begin
	∂idW = (l) -> idW(10,0.6,l);
	∂idw_∂L = derivative(∂idW, 3e-7)
end

# ╔═╡ Cell order:
# ╟─3d9fe546-3e12-11eb-3e0d-7f5e9d423e92
# ╠═647ab7f4-3e12-11eb-29a9-cd98784528f1
# ╠═39e96b2e-3fc6-11eb-339c-3dbc92012d34
# ╠═3c9843ae-3fc1-11eb-2da4-6314f7556996
# ╠═8d5934b6-3fc0-11eb-3770-97600d88d634
# ╟─80f4d512-3fbe-11eb-1548-01c553692bf4
# ╠═9635c21a-4135-11eb-2b1d-65e5ce383bce
# ╠═2c96d8d6-3fc0-11eb-1835-450c15fec5ac
# ╠═84508ff2-3fc2-11eb-2ad7-77c2953a0515
# ╠═0669423a-4136-11eb-35dd-d9e990ecb321
# ╠═f9db2f6c-4115-11eb-1a7c-9bbb7f651ec5
# ╟─0cbd8a4e-410c-11eb-07ef-1f2cce6153ed
# ╟─ed52203c-410d-11eb-2f54-e7ee56faa0de
# ╠═9de1eb8e-4105-11eb-39d3-c7d85be8f5ab
# ╠═658008b2-4109-11eb-2128-951666af5ceb
# ╟─bcdbf078-3fc2-11eb-19f9-bbddbdbca559
# ╠═76cd2dfa-4115-11eb-1c8b-095081973632
# ╠═de21a462-3fc2-11eb-3121-db02f67519cc
# ╠═f10ee5e6-3fc2-11eb-2916-3578e545b0bf
# ╟─6301e07c-4138-11eb-1606-2d9b676f2bdd
# ╠═4aff32a0-4149-11eb-2efd-a34ab9714a4e
# ╠═f5df8218-413b-11eb-385d-6dfd7ca750f8
# ╠═d54b9bee-413f-11eb-3ea1-bff788c78330
# ╠═d84ad25a-414f-11eb-2d7d-eb31e96d3af4
# ╟─693f007c-4107-11eb-194f-25b6811ceee2
# ╠═412d1876-4146-11eb-1dcc-31217a410d35
# ╠═c850e59c-4154-11eb-1931-cb3c7e18356d
