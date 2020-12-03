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
	slVds = @bind vds Slider( 0.00 : 0.01 : 1.20
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
	, yaxis = "id/W", xaxis = "gm/id"
	)

# ╔═╡ Cell order:
# ╠═9f08514e-357f-11eb-2d48-a5d0177bcc4f
# ╠═bf21b8ec-357f-11eb-023f-6b64f6e0da73
# ╠═d091d5e2-357f-11eb-385b-252f9ee49070
# ╠═ed7ac13e-357f-11eb-170b-31a27207af5f
# ╠═a002f77c-3580-11eb-0ad8-e946d85c84c7
# ╠═0282c34c-3580-11eb-28c5-e5badd2c345f
# ╠═6b97b4f0-3580-11eb-28e5-b356737b0905
# ╠═88f76a24-3580-11eb-297a-2fe3c8a9b083
