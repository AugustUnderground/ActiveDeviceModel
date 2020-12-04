### A Pluto.jl notebook ###
# v0.12.16

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
using DataFrames, JLD2, Plots, PlutoUI

# ╔═╡ 9f08514e-357f-11eb-2d48-a5d0177bcc4f
begin
	import DarkMode
    DarkMode.enable()
end

# ╔═╡ d091d5e2-357f-11eb-385b-252f9ee49070
simData = jldopen("../data/ptmn.jld") do file
	file["database"];
end;

# ╔═╡ ed7ac13e-357f-11eb-170b-31a27207af5f
simData.Vgs = round.(simData.Vgs, digits = 2);

# ╔═╡ a002f77c-3580-11eb-0ad8-e946d85c84c7
begin
	slVds = @bind vds Slider( 0.01 : 0.01 : 1.20
							, default = 0.6, show_value = true );
	slW = @bind w Slider( 5.0e-7 : 1.0e-7 : 5.0e-6
						, default = 5.0e-7, show_value = true );
	slL = @bind l Slider( 1.5e-7 : 1.0e-7 : 1.5e-6
						, default = 1.5e-7, show_value = true );
	
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
				, ["W", "L", "gm", "gds", "id", "vdsat"] ];
	dd.idw = dd.id ./ dd.W;
	dd.gmid = dd.gm ./ dd.id;
	dd.a0 = dd.gm ./ dd.gds;
end;

# ╔═╡ 42777980-3584-11eb-2f89-91d4fa8683d3
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

# ╔═╡ fcb28aca-3586-11eb-0c54-bfda0035451d
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

# ╔═╡ 439f0ba4-3587-11eb-3861-bd9ab7650012
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

# ╔═╡ 587ae052-3587-11eb-1621-2b51c52e17e8
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

# ╔═╡ 293aad98-3587-11eb-0f56-1d8144ad7e84
plot(idwgmid, idwvdsat, a0gmid, a0vdsat, layout = (2,2))

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

# ╔═╡ 88f76a24-3580-11eb-297a-2fe3c8a9b083
plot( df.gmid, df.idw
	, yscale = :log10
	, yaxis = "id/W", xaxis = "gm/id")

# ╔═╡ Cell order:
# ╠═9f08514e-357f-11eb-2d48-a5d0177bcc4f
# ╠═bf21b8ec-357f-11eb-023f-6b64f6e0da73
# ╠═d091d5e2-357f-11eb-385b-252f9ee49070
# ╠═ed7ac13e-357f-11eb-170b-31a27207af5f
# ╠═293aad98-3587-11eb-0f56-1d8144ad7e84
# ╠═a002f77c-3580-11eb-0ad8-e946d85c84c7
# ╠═092d49d4-3584-11eb-226b-bde1f2e49a22
# ╠═42777980-3584-11eb-2f89-91d4fa8683d3
# ╠═fcb28aca-3586-11eb-0c54-bfda0035451d
# ╠═439f0ba4-3587-11eb-3861-bd9ab7650012
# ╠═587ae052-3587-11eb-1621-2b51c52e17e8
# ╠═0282c34c-3580-11eb-28c5-e5badd2c345f
# ╠═6b97b4f0-3580-11eb-28e5-b356737b0905
# ╠═88f76a24-3580-11eb-297a-2fe3c8a9b083
