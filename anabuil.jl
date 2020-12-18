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
using Flux, Lazy, Printf, Plots, StatsBase, PyCall, BSON, Zygote, CUDA, mna, PlutoUI, DataFrames

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
begin
	struct MOSFET
		type
		model
		paramsX
		paramsY
		trafoX
		trafoY
	end
		
		md"""
		### Primitive Devices
		
		A _MOSFET_ is defined as a `struct` with access to the NN Model and data Transformations.
		"""
end

# ╔═╡ 2c96d8d6-3fc0-11eb-1835-450c15fec5ac
modelPath = "./model/dev-2020-12-16T16:14:05.641/ptmn90";

# ╔═╡ 84508ff2-3fc2-11eb-2ad7-77c2953a0515
function predict(mosfet, X)
	rY = ((length(size(X)) < 2) ? [X'] : X') |>
    	 mosfet.trafoX.transform |> 
         adjoint |> mosfet.model |> adjoint |>
         mosfet.trafoY.inverse_transform |> 
         adjoint
	return Float64.(rY)
end;

# ╔═╡ 145f05ce-3fc0-11eb-0378-bf8655de1384
nmos = MOSFET( "n"
			 , BSON.load(modelPath * ".bson")[:model]
			 , ["Vgs", "Vds", "Vbs", "W", "L", "eVgs", "eVds" ]
			 , [ "vth", "vdsat", "id", "gm", "gmb","gds", "fug"
			   , "cgd", "cgb", "cgs", "cds", "csb", "cdb"
			   , "idW", "gmid", "a0" ]
			 , joblib.load(modelPath * ".input")
			 , joblib.load(modelPath * ".output") );

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
	rangeEVgs = rangeVgs.^2.0;
	rangeVds = range(Vdsmin, stop = Vdsmax, length = sweepLen);
	rangeEVds = rangeVds.^0.5;
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
	, heatmap( axisL, axisGMID, A0
		   	 ; xaxis = "L", yaxis = "gm/id", zaxis = "A0"
		   	 #, zscale = :log10
		   	 , title = "Self Gain", c = :greens, legend = nothing )
	, heatmap( axisL, axisGMID, fug
		   	 ; xaxis = "L", yaxis = "gm/id", zaxis = "fug"
		   	 #, zscale = :log10
		   	 , title = "Unity Gain Frequency", c = :reds, legend = nothing )

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
	constEVgs = constVgs.^2.0;
	constVds = ones(1,sweepLen) .* Vds;
	constEVds = constVds.^0.5;
	constVbs = zeros(1,sweepLen);
end;

# ╔═╡ de21a462-3fc2-11eb-3121-db02f67519cc
begin
	Xₜ = [rangeVgs'; constVds; constVbs; constW; constL; rangeEVgs'; constEVds];
	pₜ = predict(nmos, Xₜ);
	idₜ = pₜ[first(indexin(["id"], nmos.paramsY)), :];
end;

# ╔═╡ f10ee5e6-3fc2-11eb-2916-3578e545b0bf
begin
	Xₒ = [constVgs; rangeVds'; constVbs; constW; constL; constEVgs; rangeEVds'];
	pₒ = predict(nmos, Xₒ);
	idₒ = pₜ[first(indexin(["id"], nmos.paramsY)), :];
end;

# ╔═╡ 66b68336-4105-11eb-37b0-71e1cf288f4e
begin
	rangeVL = reshape( Iterators.product(rangeVgs', rangeL') |> collect
		   		 	 , (sweepLen^2), 1);
	VL = hcat([ [o...] for o in rangeVL ]...);
	sweep = [ VL[1,:]'
			; ones(1, (sweepLen^2)) .* Vds 
			; zeros(1, (sweepLen^2))
			; ones(1, (sweepLen^2)) .* W
			; VL[2,:]' ; VL[1,:]' .^2.0
			; ones(1, (sweepLen^2)) .* (Vds^0.5) ];
	op = predict(nmos,sweep);
	opDF = DataFrame( L = VL[2,:] 
					, gmid = opp(nmos, op, "gmid")
					, vdsat = opp(nmos, op, "vdsat")
					, idW = opp(nmos, op, "idW")
					, fug = opp(nmos, op, "fug")
					, A0 = opp(nmos, op, "a0") );
	df = sort(opDF, ["L", "gmid"]);
	axisL = unique(df.L);
	axisGMID = df[ df.L .== first(axisL)
				 , "gmid" ];
	idW = reshape(df.idW, (length(axisL), length(axisGMID)));
	fug = reshape(df.fug, (length(axisL), length(axisGMID)));
	A0 = reshape(df.A0, (length(axisL), length(axisGMID)));
end;

# ╔═╡ 8bbdd950-3fc4-11eb-30d9-9f63ea017860
surface( axisL, axisGMID, idW
		   	 ; xaxis = "L", yaxis = "gm/id", zaxis = "id/W"
		   	 #, zscale = :log10
		     , title = "Current Density", c = :blues, legend = nothing )

# ╔═╡ Cell order:
# ╠═3d9fe546-3e12-11eb-3e0d-7f5e9d423e92
# ╠═647ab7f4-3e12-11eb-29a9-cd98784528f1
# ╠═39e96b2e-3fc6-11eb-339c-3dbc92012d34
# ╠═3c9843ae-3fc1-11eb-2da4-6314f7556996
# ╠═8d5934b6-3fc0-11eb-3770-97600d88d634
# ╠═80f4d512-3fbe-11eb-1548-01c553692bf4
# ╠═2c96d8d6-3fc0-11eb-1835-450c15fec5ac
# ╠═84508ff2-3fc2-11eb-2ad7-77c2953a0515
# ╠═145f05ce-3fc0-11eb-0378-bf8655de1384
# ╠═f9db2f6c-4115-11eb-1a7c-9bbb7f651ec5
# ╠═0cbd8a4e-410c-11eb-07ef-1f2cce6153ed
# ╠═ed52203c-410d-11eb-2f54-e7ee56faa0de
# ╠═9de1eb8e-4105-11eb-39d3-c7d85be8f5ab
# ╠═658008b2-4109-11eb-2128-951666af5ceb
# ╠═bcdbf078-3fc2-11eb-19f9-bbddbdbca559
# ╠═76cd2dfa-4115-11eb-1c8b-095081973632
# ╠═8bbdd950-3fc4-11eb-30d9-9f63ea017860
# ╠═693f007c-4107-11eb-194f-25b6811ceee2
# ╠═de21a462-3fc2-11eb-3121-db02f67519cc
# ╠═f10ee5e6-3fc2-11eb-2916-3578e545b0bf
# ╠═66b68336-4105-11eb-37b0-71e1cf288f4e
