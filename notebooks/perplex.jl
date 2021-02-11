### A Pluto.jl notebook ###
# v0.12.20

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
using DataFrames, StatsBase, JLD2, StatsPlots, PlutoUI, DataInterpolations, PyCall, ScikitLearn, Optim, Random, Statistics, Distributions, BSON, Flux, Zygote, CUDA, PyCall, ScikitLearn, NNlib, CSVFiles, Lazy, BoxCoxTrans, YeoJohnsonTrans, StatsBase, MLDataUtils

# ╔═╡ 31c636ac-55b8-11eb-19d6-8dc9af976a24
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
	Core.eval(Main, :(using StatsBase))
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

# ╔═╡ 0f54b05a-54ff-11eb-21e4-511378903bfe
md"""
# NMOS

## gm / id (Data Base)
"""

# ╔═╡ 5b2ea554-5a3f-11eb-3f75-2d7f8ce5a368
begin
	@sk_import preprocessing: PowerTransformer;
	@sk_import preprocessing: QuantileTransformer;
end;

# ╔═╡ 478b1cde-3e34-11eb-367b-476c0408e6c3
joblib = pyimport("joblib");

# ╔═╡ 99ba2bec-4034-11eb-045f-49b2e8eca1de
plotly();
#pyplot();

# ╔═╡ d091d5e2-357f-11eb-385b-252f9ee49070
simData = jldopen("../../data/ptmn90.jld") do file
	file["database"];
end;

# ╔═╡ ed7ac13e-357f-11eb-170b-31a27207af5f
begin
    simData.Vgs = round.(simData.Vgs, digits = 2);
    simData.Vds = round.(simData.Vds, digits = 2);
    simData.Vbs = round.(simData.Vbs, digits = 2);
end;

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
                 .& (simData.Vbs .== 0.0)
			 	 .& (simData.W .== w) )
				, ["W", "L", "gm", "gds", "id", "vdsat", "fug"] ];
	dd.idw = dd.id ./ dd.W;
	dd.gmid = dd.gm ./ dd.id;
	dd.a0 = dd.gm ./ dd.gds;
end;

# ╔═╡ a11bf6ae-6afa-11eb-0833-f5b9a6b10f70
lengths = unique(simData.L);

# ╔═╡ 24a21870-360b-11eb-1269-db94fecdb0a6
begin
	idwgmid = plot();
	#for len in 1.0e-7 : 1.0e-7 : 1.0e-6
    for len in lengths
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
    for len in lengths
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
    for len in lengths
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
    for len in lengths
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
    for len in lengths
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
    for len in lengths
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

# ╔═╡ eb065c46-5cc7-11eb-3fce-d1a566c34e82
begin
	simData.QVgs = simData.Vgs.^2.0;
	simData.EVds = ℯ.^(simData.Vds);

	paramsX = ["Vgs", "QVgs", "Vds", "EVds", "W", "L"];
	paramsY = ["vth", "vdsat", "id", "gm", "gmb", "gds", "fug"
	          , "cgd", "cgb", "cgs", "cds", "csb", "cdb" ];
	
	rawX = Matrix(simData[:, paramsX ])';
	rawY = Matrix(simData[:, paramsY ])';
end;

# ╔═╡ bc3a4716-5cd0-11eb-1939-91291bfdf530
simSample = simData[ StatsBase.sample( MersenneTwister(666)
                              		 , 1:(simData |> size |> first)
									 , pweights(simData.id)
                             		 , 666666
                            		 ; replace = false )
					, : ];

# ╔═╡ 49d1543a-5cd7-11eb-0cb9-85934736d352
begin
	bc_vgs = collect(0.0:0.01:1.2)';
	bc_qvgs = bc_vgs.^2.0;
	bc_vds = collect(0.0:0.01:1.2)';
	bc_evds = exp.(bc_vds);
	bc_len = length(bc_vgs);
	bc_W = 1.0e-6;
	bc_w = ones(1, bc_len) .* bc_W;
	bc_L = 3.0e-7;
	bc_l = ones(1, bc_len) .* bc_L;
	bc_vg = 0.6;
	bc_vgc = ones(1, bc_len) .* bc_vg;
	bc_qvgc = bc_vgc.^2.0;
	bc_vd = 0.6;
	bc_vdc = ones(1, bc_len) .* bc_vd;
	bc_evdc = exp.(bc_vdc);
	bc_vbc = zeros(1, bc_len);
	
	bc_xt = [ bc_vgs; bc_qvgs; bc_vdc; bc_evdc; bc_w; bc_l ];
	bc_xo = [ bc_vgc; bc_qvgc; bc_vds; bc_evds; bc_w; bc_l ];
end;

# ╔═╡ 21b30176-598e-11eb-0322-19cbd312896d
md"""
### Data Transformations

#### Box-Cox Transformation

$$y_{i}^{(\lambda)} = \begin{cases}
\frac{y_{i}^{\lambda} - 1}{\lambda} & \text{if }\, \lambda \neq 0 \\
\ln(y_{i}) & \text{if }\, \lambda = 0 \\
\end{cases}$$
"""

# ╔═╡ 2d96bb80-5a39-11eb-1e4f-1dd65b73dbd5
begin
	bc(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
	bc′(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);
end;

# ╔═╡ 3f07f502-5cc8-11eb-33ac-83920c977746
begin
	ut1X = StatsBase.fit(UnitRangeTransform, rawX; dims = 2, unit = true); 
	ut1Y = StatsBase.fit(UnitRangeTransform, rawY; dims = 2, unit = true);
	
	ur1X = StatsBase.transform(ut1X, rawX);
	ur1Y = StatsBase.transform(ut1Y, rawY);
	
	#coxX = hcat([ bc(rX; λ = 0.2) for rX in eachrow(ur1X)]...)';
	#coxY = hcat([ bc(rY; λ = 0.2) for rY in eachrow(ur1Y)]...)';
	coxX = bc(ur1X; λ = 0.2);
	coxY = bc(ur1Y; λ = 0.2);
	
	ut2X = StatsBase.fit(UnitRangeTransform, coxX; dims = 2, unit = true); 
	ut2Y = StatsBase.fit(UnitRangeTransform, coxY; dims = 2, unit = true);
end;

# ╔═╡ bceea132-5cd7-11eb-14ef-e70cad73f65f
begin
	us1X = StatsBase.transform(ut1X, Matrix(simSample[:,paramsX])');
	us1Y = StatsBase.transform(ut1Y, Matrix(simSample[:,paramsY])');
	
	usCX = bc(us1X .+ 1.0; λ = 0.2);
	usCY = bc(us1Y .+ 0.0; λ = 0.2);
	
	us2X = StatsBase.transform(ut2X, usCX);
	us2Y = StatsBase.transform(ut2Y, usCY);
	
	usYX = YeoJohnsonTrans.transform(us1X);
	usYY = YeoJohnsonTrans.transform(us1Y);
	
	us3X = StatsBase.standardize(UnitRangeTransform, usYX);
	us3Y = StatsBase.standardize(UnitRangeTransform, usYY);
end;

# ╔═╡ 78813f7a-5cd8-11eb-2bea-278bd4f36ec0
plot( histogram( us1Y[1,:], legend = false, title = "raw", yaxis = "vth")
	, histogram( us2Y[1,:], legend = false, title = "bct")
	, histogram( us3Y[1,:], legend = false, title = "yjt")
	, histogram( us1Y[2,:], legend = false, yaxis = "vdsat")
	, histogram( us2Y[2,:], legend = false)
	, histogram( us3Y[2,:], legend = false)
	, histogram( us1Y[3,:], legend = false, yaxis = "id")
	, histogram( usCY[3,:], legend = false)
	, histogram( us3Y[3,:], legend = false)
	, histogram( us1Y[4,:], legend = false, yaxis = "gm")
	, histogram( us2Y[4,:], legend = false)
	, histogram( us3Y[4,:], legend = false)
	, histogram( us1Y[5,:], legend = false, yaxis = "gmb")
	, histogram( us2Y[5,:], legend = false)
	, histogram( us3Y[5,:], legend = false)
	, histogram( us1Y[6,:], legend = false, yaxis = "gds")
	, histogram( us2Y[6,:], legend = false)
	, histogram( us3Y[6,:], legend = false)
	, histogram( us1Y[7,:], legend = false, yaxis = "fug")
	, histogram( us2Y[7,:], legend = false)
	, histogram( us3Y[7,:], legend = false)
	, layout = (7,3) )

# ╔═╡ 7b5eefa8-5cda-11eb-1427-e75d848c7c54
begin
	u1XT = StatsBase.transform(ut1X, bc_xt);
	cxXT = hcat([ bc(rX; λ = 0.2) for rX in eachrow(u1XT)]...)';
	u2XT = StatsBase.transform(ut2X, cxXT);
end

# ╔═╡ 297b36fc-616b-11eb-37d9-2d761bb2e287
begin
	simData.gmid = simData.gm ./ simData.id;
	simData.idW = simData.id ./ simData.W;
	simData.A0 = simData.gm ./ simData.gds;
	
	maskMatrix = Matrix(isinf.(simData) .| isnan.(simData));
	mask = (vec ∘ collect)(sum(maskMatrix, dims = 2) .== 0 );
	dataFrame = simData[mask, :];
end;

# ╔═╡ 2e70e0e8-616c-11eb-0bbb-832248e5ba56
begin
    #nsd = size(ddf)[1];
    #msd = 500000;
    #ns = @>> Lazy.range() map(x -> (nsd / (2^x)));
    #nSamp = Int.(ceil.(takewhile((x) -> x > msd, ns)));
    #tSamp = String.(take(length(nSamp), Lazy.cycle(["id", "gm"])));
    
    #dSamp = reduce( (dat, smp) -> dat[ StatsBase.sample( MersenneTwister(666)
    #                                                  , 1:(dat |> size |> first)
    #                                                  , pweights(dat[:,smp[2]])
    #                                                  , smp[1]
    #                                                  ; replace = false )
    #                                , : ]
    #             , zip(numSmpl, typSmpl); init = ddf );
    
    #asdf = dataFrame[ StatsBase.sample( MersenneTwister(666)
    #                                   , 1:size(dataFrame, 1)
    #                                   , pweights(dataFrame.id)
    #                                   , 2000000
    #                                   ; replace = false
    #                                   , ordered = false )
    #                 , : ];

    sdf = dataFrame[dataFrame.Vds .>= (dataFrame.Vgs .- dataFrame.vth), :];
    sSamp = sdf[ StatsBase.sample( MersenneTwister(666)
                                 , 1:size(sdf, 1)
                                 , pweights(sdf.id)
                                 , 3000000
                                 ; replace = false
                                 , ordered = false )
               , : ];

    tdf = dataFrame[dataFrame.Vds .<= (dataFrame.Vgs .- dataFrame.vth), :];
    tSamp = tdf[ StatsBase.sample( MersenneTwister(666)
                                 , 1:size(tdf, 1)
                                 #, pweights(tdf.id)
                                 , 1000000
                                 ; replace = false
                                 , ordered = false )
               , : ];

    asdf = shuffleobs(vcat(sSamp, tSamp));
end;

# ╔═╡ a4ef07d2-616b-11eb-27e3-f77fb919898d
plot( histogram(bc(asdf.id); yaxis = "id", title = "Box Cox'ed")
	, histogram(asdf.id; yaxis = "id", title = "Raw")
	, histogram(bc(asdf.Vds); yaxis = "Vds", xaxis = "Value") 
	, histogram(asdf.Vds; yaxis = "Vds", xaxis = "Value") 
 	; layout = (2,2)
	, legend = false)

# ╔═╡ 5ba2fb94-5985-11eb-1710-932d14cb2c51
md"""
# Comparison
"""

# ╔═╡ 19e00e34-55b8-11eb-2ecd-3398e288598a
begin	
	modelPathₙ = "../model/ptmn90-2021-01-28T13:08:19.923/ptmn90";
	modelFileₙ = modelPathₙ * ".bson";
	modelₙ = BSON.load(modelFileₙ);
	φₙ = modelₙ[:model];
	paramsXₙ = modelₙ[:paramsX];
	paramsYₙ = modelₙ[:paramsY];
	utXₙ = modelₙ[:utX];
	utYₙ = modelₙ[:utY];
	
	function predictₙ(X)
		X′ = StatsBase.transform(utXₙ, X);
		Y′ = φₙ(X′);
		Y = StatsBase.reconstruct(utYₙ, Y′);
  		return Float64.(Y)
	end;
end;

# ╔═╡ fae88ea6-6157-11eb-005a-b7fe5ab2bb16
begin	
	modelPathₚ = "../model/ptmp90-2021-01-28T11:53:55.782/ptmp90";
	modelFileₚ = modelPathₚ * ".bson";
	modelₚ = BSON.load(modelFileₚ);
	φₚ = modelₚ[:model];
	paramsXₚ = modelₚ[:paramsX];
	paramsYₚ = modelₚ[:paramsY];
	utXₚ = modelₚ[:utX];
	utYₚ = modelₚ[:utY];
	
	function predictₚ(X)
		X′ = StatsBase.transform(utXₚ, X);
		Y′ = φₚ(X′);
		Y = StatsBase.reconstruct(utYₚ, Y′);
  		return Float64.(Y)
	end;
end;

# ╔═╡ 917a3ff2-55bb-11eb-36f3-fb1d62827973
begin
	vgsC = collect(0.0:0.01:1.2)';
	qvgsC = vgsC.^2.0;
	vdsC = collect(0.0:0.01:1.2)';
	evdsC = exp.(vdsC);
end;

# ╔═╡ 9e3dee36-55bc-11eb-355f-2f28faf37480
begin
	slWC = @bind wC Slider( 1.0e-6 : 2.5e-7 : 4.75e-6
						, default = 1.0e-6, show_value = true );
	slLC = @bind lC Slider( 3.0e-7 : 1.5e-7 : 3e-6
						, default = 3.0e-7, show_value = true );
	
	md"""
	W = $(slWC)
	
	L = $(slLC)
	"""
end

# ╔═╡ 0d74596a-55be-11eb-0dea-ef5d6ab219c6
begin
	dat = simData[ ( (simData.W  .== wC)
                  .& (simData.Vbs .== 0.0)
				  .& (simData.L .== lC) ) 
		   		 , ["id", "Vgs", "Vds" ] ];
	len = size(dat) |> first;
	
	sweepₙ = [ dat.Vgs'
			 ; dat.Vgs' .^ 2.0
			 ; dat.Vds'
			 ; exp.(dat.Vds)'
			 ; fill(wC, 1, len) 
			 ; fill(lC, 1, len) ];
	
	yₙ = predictₙ(sweepₙ);
	yₚ = predictₚ(sweepₙ);
	
	idₚₙ = reshape( yₙ[first(indexin(["id"], paramsYₙ)), :]
				 , (Int(sqrt(len)), Int(sqrt(len))));
	idₜₙ = reshape( dat.id
				 , (Int(sqrt(len)), Int(sqrt(len))));	
	
	idₚₚ = reshape( yₚ[first(indexin(["id"], paramsYₚ)), :]
				 , (Int(sqrt(len)), Int(sqrt(len))));
end;

# ╔═╡ 57ff103c-55bf-11eb-294a-d5ce2b0557c8
begin
	surface(unique(dat.Vgs), unique(dat.Vds), idₜₙ; c = :blues, legend = false);
	surface!(unique(dat.Vgs), unique(dat.Vds), idₚₙ; c = :reds, legend = false)
end

# ╔═╡ 64509de8-6158-11eb-399c-c5671efffd88
#surface(unique(dat.Vgs), unique(dat.Vds), idₚₚ; c = :reds, legend = false)

# ╔═╡ c059152c-5a7b-11eb-062c-9f68eca827dc
md"""
## Derivatives

While the absolute values of PREDICT data points are not technology independent, the general functionality of a Transistor _is_. E.g. by increasing the _width_, a larger current can fit through the device, regardless of technology node.
"""

# ╔═╡ faad964e-5a7b-11eb-30e5-7552e886738c
begin
	ddL = dd[(dd.L .== rand(unique(dd.L))),:]
	sort!(ddL, [:gmid])
	#L = diff(ddL.L);
	∂gmid = diff(ddL.gmid);
	#∂idW′∂L = diff(ddL.idw) ./ ∂L;
	#∂A0′∂L = diff(ddL.a0) ./ ∂L;
	∂idW′∂gmid = diff(ddL.idw) ./ ∂gmid;
	∂A0′∂gmid = diff(ddL.a0) ./ ∂gmid;
end;

# ╔═╡ f9a4397e-5a80-11eb-3843-2db5c69de322
begin
	plot(ddL.gmid, ddL.idw; label = "idW/(gm/id)", xaxis = "gm/id", yaxis = "id/W");
	plot!(ddL.gmid[2:end], ∂idW′∂gmid; label = "∂idW/∂(gm/id)", title = "∂gm/id")
end

# ╔═╡ Cell order:
# ╠═9f08514e-357f-11eb-2d48-a5d0177bcc4f
# ╠═5d549288-3a0c-11eb-0ac3-595f54266cb3
# ╟─0f54b05a-54ff-11eb-21e4-511378903bfe
# ╠═bf21b8ec-357f-11eb-023f-6b64f6e0da73
# ╠═31c636ac-55b8-11eb-19d6-8dc9af976a24
# ╠═5b2ea554-5a3f-11eb-3f75-2d7f8ce5a368
# ╠═478b1cde-3e34-11eb-367b-476c0408e6c3
# ╠═99ba2bec-4034-11eb-045f-49b2e8eca1de
# ╠═d091d5e2-357f-11eb-385b-252f9ee49070
# ╠═ed7ac13e-357f-11eb-170b-31a27207af5f
# ╟─293aad98-3587-11eb-0f56-1d8144ad7e84
# ╟─a002f77c-3580-11eb-0ad8-e946d85c84c7
# ╠═092d49d4-3584-11eb-226b-bde1f2e49a22
# ╠═a11bf6ae-6afa-11eb-0833-f5b9a6b10f70
# ╠═24a21870-360b-11eb-1269-db94fecdb0a6
# ╠═c6232b50-360b-11eb-18a2-39bdc25fb03b
# ╠═cff6fad6-360b-11eb-3e9b-a7cf6a270f8f
# ╠═d1b49f7e-360b-11eb-2b4d-b5a6ab46505e
# ╠═d34046d6-360b-11eb-31cd-6378f8c1729c
# ╠═d46c5e3c-360b-11eb-3ab7-9dc5eeb107d6
# ╠═0282c34c-3580-11eb-28c5-e5badd2c345f
# ╠═6b97b4f0-3580-11eb-28e5-b356737b0905
# ╟─799725d6-4034-11eb-2f62-91ef4cc5693c
# ╠═eb065c46-5cc7-11eb-3fce-d1a566c34e82
# ╠═3f07f502-5cc8-11eb-33ac-83920c977746
# ╠═bc3a4716-5cd0-11eb-1939-91291bfdf530
# ╠═bceea132-5cd7-11eb-14ef-e70cad73f65f
# ╠═78813f7a-5cd8-11eb-2bea-278bd4f36ec0
# ╠═49d1543a-5cd7-11eb-0cb9-85934736d352
# ╠═7b5eefa8-5cda-11eb-1427-e75d848c7c54
# ╟─21b30176-598e-11eb-0322-19cbd312896d
# ╠═2d96bb80-5a39-11eb-1e4f-1dd65b73dbd5
# ╠═297b36fc-616b-11eb-37d9-2d761bb2e287
# ╠═2e70e0e8-616c-11eb-0bbb-832248e5ba56
# ╠═a4ef07d2-616b-11eb-27e3-f77fb919898d
# ╟─5ba2fb94-5985-11eb-1710-932d14cb2c51
# ╠═19e00e34-55b8-11eb-2ecd-3398e288598a
# ╠═fae88ea6-6157-11eb-005a-b7fe5ab2bb16
# ╠═917a3ff2-55bb-11eb-36f3-fb1d62827973
# ╠═0d74596a-55be-11eb-0dea-ef5d6ab219c6
# ╠═57ff103c-55bf-11eb-294a-d5ce2b0557c8
# ╟─9e3dee36-55bc-11eb-355f-2f28faf37480
# ╠═64509de8-6158-11eb-399c-c5671efffd88
# ╟─c059152c-5a7b-11eb-062c-9f68eca827dc
# ╠═faad964e-5a7b-11eb-30e5-7552e886738c
# ╟─f9a4397e-5a80-11eb-3843-2db5c69de322
