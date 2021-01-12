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
using DataFrames, StatsBase, JLD2, StatsPlots, PlutoUI, DataInterpolations, PyCall, ScikitLearn, Optim, Random, Statistics, Distributions

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
simData = jldopen("../../data/ptmn90.jld") do file
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

# ╔═╡ 34f10fea-5192-11eb-05f5-edab6dd10104
md"""
### All Data
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
		, layout = lay , legend = false )
end

# ╔═╡ 719bc782-5192-11eb-22bd-f157baf48c57
md"""
### Noramlly Distributed Sample
"""

# ╔═╡ 58d6a226-5188-11eb-3b6a-6598007e3fda
begin
	σ = Statistics.var(Matrix(simData); dims = 1) |> collect |> vec;
	μ = Statistics.mean(Matrix(simData); dims = 1) |> collect |> vec;
	normal = MvNormal(σ, μ);
	uniform = MersenneTwister(666)
end;

# ╔═╡ 8a6d0320-5192-11eb-1e08-4d1fabfa99f8
begin
	numSamples = 10000;
	normSamples = rand(normal, numSamples);
	unifSamples = StatsBase.sample( uniform
                              	  , 1:(simData |> size |> first)
                             	  , numSamples
                             	  ; replace = false);

	normData = rename!(convert(DataFrame, normSamples'), names(simData));
	unifData = simData[unifSamples, :];
end;

# ╔═╡ 82af7f46-518d-11eb-0d12-0355938010c9
plot( histogram(unifData.id; title = "Uniform")
	, histogram(normData.id; title = "Normal")
	; layout = (1, 2), legend = false )

# ╔═╡ Cell order:
# ╠═9f08514e-357f-11eb-2d48-a5d0177bcc4f
# ╠═5d549288-3a0c-11eb-0ac3-595f54266cb3
# ╠═472a5f78-3a1c-11eb-31da-9fe4b67106e4
# ╠═bf21b8ec-357f-11eb-023f-6b64f6e0da73
# ╠═99ba2bec-4034-11eb-045f-49b2e8eca1de
# ╠═478b1cde-3e34-11eb-367b-476c0408e6c3
# ╠═d091d5e2-357f-11eb-385b-252f9ee49070
# ╠═ed7ac13e-357f-11eb-170b-31a27207af5f
# ╟─293aad98-3587-11eb-0f56-1d8144ad7e84
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
# ╟─799725d6-4034-11eb-2f62-91ef4cc5693c
# ╟─34f10fea-5192-11eb-05f5-edab6dd10104
# ╠═8767dc1e-4034-11eb-2c11-91b94e7644b8
# ╟─719bc782-5192-11eb-22bd-f157baf48c57
# ╠═58d6a226-5188-11eb-3b6a-6598007e3fda
# ╠═8a6d0320-5192-11eb-1e08-4d1fabfa99f8
# ╠═82af7f46-518d-11eb-0d12-0355938010c9
