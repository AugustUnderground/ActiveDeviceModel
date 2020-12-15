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

# ╔═╡ bf21b8ec-357f-11eb-023f-6b64f6e0da73
using DataFrames, StatsBase, JLD2, Plots, PlutoUI, DataInterpolations, Flux, Zygote, CUDA, BSON, PyCall, ScikitLearn, NNlib

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
			 	       , yaxis = "id/W", xaxis = "gm/id" );
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
			 	   		, yaxis = "id/W", xaxis = "vdsat" );
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
			 	      , yaxis = "A0", xaxis = "gm/id" );
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
			 	   	   , yaxis = "A0", xaxis = "vdsat" );
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
			 	   	  , yaxis = "fug", xaxis = "gmid" );
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
			 	   	   , yaxis = "fug", xaxis = "vdsat" );
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

# ╔═╡ 3cf1f458-3a1c-11eb-2d51-a70a21c10295
md"""
## gm / id (Neural Network)
"""

# ╔═╡ 49e8abac-3e18-11eb-28ca-f9af0718950d
begin
	modelPath = "./model/dev-2020-12-14T17:50:21.395/ptmn90";
	modelFile = modelPath * ".bson";
	trafoInFile = modelPath * ".input";
	trafoOutFile = modelPath * ".output";
	model = BSON.load(modelFile);
	φ = model[:model];
	trafoX = joblib.load(trafoInFile);
	trafoY = joblib.load(trafoOutFile);
end;

# ╔═╡ 219e21a4-3e1d-11eb-2a02-fd152e843650
function predict(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoX.transform |> 
         adjoint |> φ |> adjoint |>
         trafoY.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end

# ╔═╡ 910b6bfa-3e36-11eb-34f6-4d9bf3df8188


# ╔═╡ f2dc08a6-3a1e-11eb-08b3-81a2ce43c86a
begin
	scVds = @bind cvds Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	scVgs = @bind cvgs Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	scW = @bind cw Slider( 7.5e-7 : 1.0e-7 : 5.0e-6
						, default = 5.0e-7, show_value = true );
	scL = @bind cl Slider( 1.5e-7 : 1.0e-7 : 1.5e-6
						, default = 1.5e-7, show_value = true );
	
	md"""
	vds = $(scVds)
	
	W = $(scW)
	
	L = $(scL)
	"""
end

# ╔═╡ 99fb92e4-3e1d-11eb-3120-7b09e7d9a257
begin
	paramsXY = names(simData);
	paramsX = filter((p) -> isuppercase(first(p)), paramsXY);
	paramsY = filter((p) -> !in(p, paramsX), paramsXY);
end;

# ╔═╡ 5d9312be-3e1d-11eb-184e-6fc51d067282
begin
	vg = 0.0:0.01:1.2;
	vd = 0.0:0.01:1.2;
end;

# ╔═╡ ac7b8cb8-3e35-11eb-2e5b-234637084d4e
paramsX

# ╔═╡ c60af316-3e1d-11eb-238c-d5ef097d9875
# Input matrix for φ according to paramsX
dp = [ collect(vg)'
    ; repeat([cvds], 121)'
    ; zeros(1, 121)
    ; repeat([cw], 121)'
    ; repeat([cl], 121)' ]

# ╔═╡ f67a824c-3e35-11eb-0d62-215d8f7aaeca
opp = predict(dp)

# ╔═╡ b5eaefe0-3e36-11eb-31d4-633a724d1dd9
begin
	nn_gm = opp[first(indexin(["gm"], paramsY)), :];
	nn_id = opp[first(indexin(["id"], paramsY)), :];
	nn_idwgmid = plot( (nn_gm ./ nn_id)
			 	   	 , (nn_id ./ cw)
			 	   	 , yscale = :log10
				     , legend = false
			 	   	 , yaxis = "id/W", xaxis = "gm/id" );
end

# ╔═╡ Cell order:
# ╠═9f08514e-357f-11eb-2d48-a5d0177bcc4f
# ╠═5d549288-3a0c-11eb-0ac3-595f54266cb3
# ╠═472a5f78-3a1c-11eb-31da-9fe4b67106e4
# ╠═bf21b8ec-357f-11eb-023f-6b64f6e0da73
# ╠═5b9d18dc-3e19-11eb-03e9-9f231903bd84
# ╠═478b1cde-3e34-11eb-367b-476c0408e6c3
# ╠═d091d5e2-357f-11eb-385b-252f9ee49070
# ╠═ed7ac13e-357f-11eb-170b-31a27207af5f
# ╠═293aad98-3587-11eb-0f56-1d8144ad7e84
# ╠═a002f77c-3580-11eb-0ad8-e946d85c84c7
# ╠═092d49d4-3584-11eb-226b-bde1f2e49a22
# ╠═24a21870-360b-11eb-1269-db94fecdb0a6
# ╠═c6232b50-360b-11eb-18a2-39bdc25fb03b
# ╠═cff6fad6-360b-11eb-3e9b-a7cf6a270f8f
# ╠═d1b49f7e-360b-11eb-2b4d-b5a6ab46505e
# ╠═d34046d6-360b-11eb-31cd-6378f8c1729c
# ╠═d46c5e3c-360b-11eb-3ab7-9dc5eeb107d6
# ╠═0282c34c-3580-11eb-28c5-e5badd2c345f
# ╠═6b97b4f0-3580-11eb-28e5-b356737b0905
# ╠═3cf1f458-3a1c-11eb-2d51-a70a21c10295
# ╠═49e8abac-3e18-11eb-28ca-f9af0718950d
# ╠═219e21a4-3e1d-11eb-2a02-fd152e843650
# ╠═910b6bfa-3e36-11eb-34f6-4d9bf3df8188
# ╠═f2dc08a6-3a1e-11eb-08b3-81a2ce43c86a
# ╠═99fb92e4-3e1d-11eb-3120-7b09e7d9a257
# ╠═5d9312be-3e1d-11eb-184e-6fc51d067282
# ╠═ac7b8cb8-3e35-11eb-2e5b-234637084d4e
# ╠═c60af316-3e1d-11eb-238c-d5ef097d9875
# ╠═f67a824c-3e35-11eb-0d62-215d8f7aaeca
# ╠═b5eaefe0-3e36-11eb-31d4-633a724d1dd9
