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

# ╔═╡ bf21b8ec-357f-11eb-023f-6b64f6e0da73
using DataFrames, StatsBase, JLD2, Plots, PlutoUI, DataInterpolations, Flux, Zygote, CUDA, BSON, PyCall, ScikitLearn, NNlib, FiniteDifferences, Optim

# ╔═╡ 5b9d18dc-3e19-11eb-03e9-9f231903bd84
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
end

# ╔═╡ 9f08514e-357f-11eb-2d48-a5d0177bcc4f
#begin
#	import DarkMode
#	config = Dict( "tabSize" => 4
#				 , "keyMap" => "vim" );
#	DarkMode.enable( theme = "ayu-mirage"
#				   , cm_config = config	)
#end

# ╔═╡ 5d549288-3a0c-11eb-0ac3-595f54266cb3
#DarkMode.themes

# ╔═╡ 472a5f78-3a1c-11eb-31da-9fe4b67106e4
md"""
## gm / id (Data Base)
"""

# ╔═╡ 99ba2bec-4034-11eb-045f-49b2e8eca1de
plotly();

# ╔═╡ 478b1cde-3e34-11eb-367b-476c0408e6c3
joblib = pyimport("joblib");

# ╔═╡ d091d5e2-357f-11eb-385b-252f9ee49070
simData = jldopen("../data/ptmn90.jld") do file
	file["database"];
end;

# ╔═╡ ed7ac13e-357f-11eb-170b-31a27207af5f
simData.Vgs = round.(simData.Vgs, digits = 2);

# ╔═╡ a002f77c-3580-11eb-0ad8-e946d85c84c7
begin
	slVds = @bind vds Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	slW = @bind w Slider( 1.0e-6 : 2.5e-7 : 5.0e-6
						, default = 1.0e-6, show_value = true );
	slL = @bind l Slider( 3.0e-7 : 1.0e-7 : 1.5e-6
						, default = 3.0e-7, show_value = true );
	
	md"""
	vds = $(slVds)
	
	W = $(slW)
	
	L = $(slL)
	"""
end

# ╔═╡ 092d49d4-3584-11eb-226b-bde1f2e49a22
begin
	dd = simData[ ( (simData.Vds .== vds)
			 	 .& (simData.W .== w) )
				, ["W", "L", "gm", "gds", "id", "vdsat", "fug"] ];
	dd.idw = dd.id ./ dd.W;
	dd.gmid = dd.gm ./ dd.id;
	dd.a0 = dd.gm ./ dd.gds;
end;

# ╔═╡ 24a21870-360b-11eb-1269-db94fecdb0a6
begin
	idwgmid = plot();
	for len in 1.5e-7 : 1.0e-7 : 1.5e-6
		idwgmid = plot!( dd[dd.L .== len, "gmid"]
			 	   	   , dd[dd.L .== len, "idw"]
			 	   	   , yscale = :log10
				   	   , lab = "L = " *string(len)
				       , legend = false
			 	       , yaxis = "id/W", title = "gm/id" );
	end;
	idwgmid
end;

# ╔═╡ c6232b50-360b-11eb-18a2-39bdc25fb03b
begin
	idwvdsat = plot();
	for len in 1.5e-7 : 1.0e-7 : 1.5e-6
		idwvdsat = plot!( dd[dd.L .== len, "vdsat"]
			 	       	, dd[dd.L .== len, "idw"]
			 	   		, yscale = :log10
				   		, lab = "L = " *string(len)
				   		, legend = false
			 	   		, yaxis = "id/W", title = "vdsat" );
	end;
	idwvdsat
end;

# ╔═╡ cff6fad6-360b-11eb-3e9b-a7cf6a270f8f
begin
	a0gmid = plot();
	for len in 1.5e-7 : 1.0e-7 : 1.5e-6
		a0gmid = plot!( dd[dd.L .== len, "gmid"]
			 	   	  , dd[dd.L .== len, "a0"]
			 	      , yscale = :log10
				      , lab = "L = " *string(len)
				      , legend = false
			 	      , yaxis = "A0" );
	end;
	a0gmid
end;

# ╔═╡ d1b49f7e-360b-11eb-2b4d-b5a6ab46505e
begin
	a0vdsat = plot();
	for len in 1.5e-7 : 1.0e-7 : 1.5e-6
		a0vdsat = plot!( dd[dd.L .== len, "vdsat"]
			 	   	   , dd[dd.L .== len, "a0"]
			 	   	   , yscale = :log10
				   	   , lab = "L = " *string(len)
				   	   , legend = false
			 	   	   , yaxis = "A0" );
	end;
	a0vdsat
end;

# ╔═╡ d34046d6-360b-11eb-31cd-6378f8c1729c
begin
	ftgmid = plot();
	for len in 1.5e-7 : 1.0e-7 : 1.5e-6
		ftgmid = plot!( dd[dd.L .== len, "gmid"]
			 	   	  , dd[dd.L .== len, "fug"]
			 	   	  , yscale = :log10
				   	  , lab = "L = " *string(len)
				   	  , legend = false
			 	   	  , yaxis = "fug" );
	end;
	ftgmid
end;

# ╔═╡ d46c5e3c-360b-11eb-3ab7-9dc5eeb107d6
begin
	ftvdsat = plot();
	for len in 1.5e-7 : 1.0e-7 : 1.5e-6
		ftvdsat = plot!( dd[dd.L .== len, "vdsat"]
			 	   	   , dd[dd.L .== len, "fug"]
			 	   	   , yscale = :log10
				   	   , lab = "L = " *string(len)
				   	   , legend = false
			 	   	   , yaxis = "fug" );
	end;
	ftvdsat
end;

# ╔═╡ 293aad98-3587-11eb-0f56-1d8144ad7e84
plot(idwgmid, idwvdsat, a0gmid, a0vdsat, ftgmid, ftvdsat, layout = (3,2))

# ╔═╡ 0282c34c-3580-11eb-28c5-e5badd2c345f
df = simData[ ( (simData.Vds .== vds)
			 .& (simData.L .== l)
			 .& (simData.W .== w) )
			, ["W", "L", "gm", "gds", "id", "vdsat"] ];

# ╔═╡ 6b97b4f0-3580-11eb-28e5-b356737b0905
begin
	df.idw = df.id ./ df.W;
	df.gmid = df.gm ./ df.id;
	df.a0 = df.gm ./ df.gds;
end;

# ╔═╡ 799725d6-4034-11eb-2f62-91ef4cc5693c
md"""
## Statistics
"""

# ╔═╡ 8767dc1e-4034-11eb-2c11-91b94e7644b8
begin
	lay = @layout [ a b
				   [c 
					d 
					e] 
				  ]
	plot( histogram((simData.gm ./ simData.id), title = "gm/id")
		, histogram((simData.vdsat), title = "vdsat")
		, histogram((simData.id ./ simData.W), title = "idW")
		, histogram((simData.fug), title = "fug")
		, histogram((simData.gm ./ simData.gds), title = "A0") 
		, layout = lay , legend = false)
end

# ╔═╡ 115372aa-4038-11eb-2828-6b688d7e0dae
begin
	l1 = @layout [ a b
				   [c 
					d 
					e] 
				  ]
	plot( histogram(log.(simData.gm ./ simData.id), title = "gm/id")
		, histogram(log.(simData.vdsat), title = "vdsat")
		, histogram(log.(simData.id ./ simData.W), title = "idW")
		, histogram(log.(simData.fug), title = "fug")
		, histogram(log.(simData.gm ./ simData.gds), title = "A0") 
		, layout = l1 , legend = false);
end

# ╔═╡ 3cf1f458-3a1c-11eb-2d51-a70a21c10295
md"""
## gm / id (Neural Network)
"""

# ╔═╡ 48a26e26-4e72-11eb-222d-090029af4981
begin	
	modelPath = "./model/dev-2021-01-04T10:33:31.124/ptmn90";
	modelFile = modelPath * ".bson";
	trafoInFile = modelPath * ".input";
	trafoOutFile = modelPath * ".output";
	model = BSON.load(modelFile);
	φ = model[:model];
	trafoX = joblib.load(trafoInFile);
	trafoY = joblib.load(trafoOutFile);
	paramsX = model[:paramsX];
	paramsY = model[:paramsY];
end;

# ╔═╡ 219e21a4-3e1d-11eb-2a02-fd152e843650
function predict(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoX.transform |> 
         adjoint |> φ |> adjoint |>
         trafoY.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end;

# ╔═╡ b8fd5de8-403a-11eb-0312-6dc9000006ea
nmos = (vgs, vds, vbs, w, l) -> predict([vgs, vds, vbs, w, l, vgs^2, exp(vds)]);

# ╔═╡ f2dc08a6-3a1e-11eb-08b3-81a2ce43c86a
begin
	scVds = @bind cvds Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	scVgs = @bind cvgs Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	scW = @bind cw Slider( 7.5e-7 : 1.0e-7 : 5.0e-6
						, default = 5.0e-7, show_value = true );
	scL = @bind cl Slider( 3e-7 : 1.0e-7 : 1.5e-6
						, default = 3e-7, show_value = true );
	
	md"""
	`Vds` = $(scVds) `Vgs` = $(scVgs)
	
	`W` = $(scW) `L` = $(scL)
	"""
end

# ╔═╡ 5d9312be-3e1d-11eb-184e-6fc51d067282
begin
	vg = collect(0.0:0.01:1.2)';
	qvgs = vg.^2.0;
	vd = collect(0.0:0.01:1.2)';
	evds = exp.(vd);
	le = length(vg);
	vgc = ones(1,le) .* cvgs;
	qvgc = vgc.^2.0;
	vdc = ones(1,le) .* cvds;
	evdc = exp.(vdc);
	len = ones(1,le) .* cl;
	wid = ones(1,le) .* cw;
	vbc = zeros(1,le);
end;

# ╔═╡ f67a824c-3e35-11eb-0d62-215d8f7aaeca
begin
	dp = [ vg; vdc; vbc; wid; len; qvgs; evdc ];
	opp = predict(dp);
end;

# ╔═╡ 5a73a78a-406c-11eb-32e5-356b9cf0bf24
begin
	opv = reshape( Iterators.product(vg', vd') |> collect
		   		 , ((length(vg) * length(vd)), 1));
	vgd = hcat([ [o...] for o in opv ]...);
	slen = size(vgd)[2]
	sweep = [ vgd[2,:]'
			; vgd[1,:]'
			; zeros(1, slen)
			; ones(1, slen) .* cw 
			; ones(1, slen) .* cl
			; vgd[2,:]'.^(2.0)
			; exp.(vgd[1,:])' ];
		id = reshape( predict(sweep)[first(indexin(["id"], paramsY)), :]
					, (Int(sqrt(slen)), Int(sqrt(slen))));
end;

# ╔═╡ f36de5b2-4074-11eb-2086-b987caf75bdd
surface(vg',vd',id'; c = :blues, xaxis = "Vds", yaxis = "Vgs", zaxis = "Id")

# ╔═╡ 22fb2746-50e9-11eb-1c53-9dbe2b11e979
md"""
## Reduced Model Analysis
"""

# ╔═╡ fc7096d8-50e8-11eb-13bc-bda8818325be
begin	
	RmodelPath = "./model/dev-2021-01-07T15:22:24.59/ptmn90";
	RmodelFile = RmodelPath * ".bson";
	RtrafoInFile = RmodelPath * ".input";
	RtrafoOutFile = RmodelPath * ".output";
	Rmodel = BSON.load(RmodelFile);
	φᵣ = Rmodel[:model];
	RtrafoX = joblib.load(RtrafoInFile);
	RtrafoY = joblib.load(RtrafoOutFile);
	RparamsX = Rmodel[:paramsX];
	RparamsY = Rmodel[:paramsY];
end;

# ╔═╡ 3640d672-50e9-11eb-20f7-bf1258644657
function Rpredict(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         RtrafoX.transform |> 
         adjoint |> φᵣ |> adjoint |>
         RtrafoY.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end;

# ╔═╡ 45bda8e6-50e9-11eb-025a-7dee332124ca
nmosᵣ = (vgs, vds, w, l) -> Rpredict([vgs, vds, w, l]);

# ╔═╡ 9b277e56-50e9-11eb-2ab3-31abcf8b0293
begin
	Ropv = reshape( Iterators.product(vg', vd') |> collect
		   		  , ((length(vg) * length(vd)), 1));
	Rvgd = hcat([ [o...] for o in Ropv ]...);
	Rslen = size(Rvgd)[2]
	Rsweep = [ Rvgd[2,:]'
			 ; Rvgd[1,:]'
			 ; ones(1, Rslen) .* cw 
			 ; ones(1, Rslen) .* cl ];
		idᵣ = reshape( Rpredict(Rsweep)[first(indexin(["id"], RparamsY)), :]
					 , (Int(sqrt(Rslen)), Int(sqrt(Rslen))));
end;

# ╔═╡ 8ef3debc-50ea-11eb-2cb1-1de0e0d18dd2
surface(vg',vd',idᵣ'; c = :greens, xaxis = "Vds", yaxis = "Vgs", zaxis = "Id")

# ╔═╡ Cell order:
# ╠═9f08514e-357f-11eb-2d48-a5d0177bcc4f
# ╠═5d549288-3a0c-11eb-0ac3-595f54266cb3
# ╠═472a5f78-3a1c-11eb-31da-9fe4b67106e4
# ╠═bf21b8ec-357f-11eb-023f-6b64f6e0da73
# ╠═99ba2bec-4034-11eb-045f-49b2e8eca1de
# ╠═5b9d18dc-3e19-11eb-03e9-9f231903bd84
# ╠═478b1cde-3e34-11eb-367b-476c0408e6c3
# ╠═d091d5e2-357f-11eb-385b-252f9ee49070
# ╠═ed7ac13e-357f-11eb-170b-31a27207af5f
# ╠═293aad98-3587-11eb-0f56-1d8144ad7e84
# ╟─a002f77c-3580-11eb-0ad8-e946d85c84c7
# ╠═092d49d4-3584-11eb-226b-bde1f2e49a22
# ╠═24a21870-360b-11eb-1269-db94fecdb0a6
# ╠═c6232b50-360b-11eb-18a2-39bdc25fb03b
# ╠═cff6fad6-360b-11eb-3e9b-a7cf6a270f8f
# ╠═d1b49f7e-360b-11eb-2b4d-b5a6ab46505e
# ╠═d34046d6-360b-11eb-31cd-6378f8c1729c
# ╠═d46c5e3c-360b-11eb-3ab7-9dc5eeb107d6
# ╠═0282c34c-3580-11eb-28c5-e5badd2c345f
# ╠═6b97b4f0-3580-11eb-28e5-b356737b0905
# ╠═799725d6-4034-11eb-2f62-91ef4cc5693c
# ╟─8767dc1e-4034-11eb-2c11-91b94e7644b8
# ╟─115372aa-4038-11eb-2828-6b688d7e0dae
# ╟─3cf1f458-3a1c-11eb-2d51-a70a21c10295
# ╠═48a26e26-4e72-11eb-222d-090029af4981
# ╠═219e21a4-3e1d-11eb-2a02-fd152e843650
# ╠═b8fd5de8-403a-11eb-0312-6dc9000006ea
# ╠═f36de5b2-4074-11eb-2086-b987caf75bdd
# ╟─f2dc08a6-3a1e-11eb-08b3-81a2ce43c86a
# ╠═5d9312be-3e1d-11eb-184e-6fc51d067282
# ╠═f67a824c-3e35-11eb-0d62-215d8f7aaeca
# ╠═5a73a78a-406c-11eb-32e5-356b9cf0bf24
# ╟─22fb2746-50e9-11eb-1c53-9dbe2b11e979
# ╠═8ef3debc-50ea-11eb-2cb1-1de0e0d18dd2
# ╠═fc7096d8-50e8-11eb-13bc-bda8818325be
# ╠═3640d672-50e9-11eb-20f7-bf1258644657
# ╠═45bda8e6-50e9-11eb-025a-7dee332124ca
# ╠═9b277e56-50e9-11eb-2ab3-31abcf8b0293
