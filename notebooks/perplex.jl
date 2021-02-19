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

# ╔═╡ bc3a4716-5cd0-11eb-1939-91291bfdf530
begin
    simSample = simData[ StatsBase.sample( MersenneTwister(666)
                                         , 1:(simData |> size |> first)
                                         , pweights(simData.id)
                                         , 666666
                                         ; replace = false )
                       , : ];
    rawSample = simData[ StatsBase.sample( MersenneTwister(666)
                                         , 1:(simData |> size |> first)
                                         , 666666
                                         ; replace = false )
                       , : ];
end;

# ╔═╡ bceea132-5cd7-11eb-14ef-e70cad73f65f
begin
	rs1Y = StatsBase.transform(ut1Y, Matrix(rawSample[:,paramsY])');

	us1X = StatsBase.transform(ut1X, Matrix(simSample[:,paramsX])');
	us1Y = StatsBase.transform(ut1Y, Matrix(simSample[:,paramsY])');
	
	usCX = bc(us1X .+ 1.0; λ = 0.2);
	usCY = bc(us1Y .+ 0.0; λ = 0.2);
	
	us2X = StatsBase.transform(ut2X, usCX);
	us2Y = StatsBase.transform(ut2Y, usCY);
	
	#usYX = YeoJohnsonTrans.transform(us1X);
	#usYY = YeoJohnsonTrans.transform(us1Y);
	
	#us3X = StatsBase.standardize(UnitRangeTransform, usYX);
	#us3Y = StatsBase.standardize(UnitRangeTransform, usYY);

    usQX = QuantileTransformer( output_distribution = "uniform"
                              , random_state = 666 );
    usQY = QuantileTransformer( output_distribution = "uniform"
                              , random_state = 666 );

    us3X = us1X |> adjoint |> usQX.fit_transform |> adjoint;
    us3Y = us1Y |> adjoint |> usQY.fit_transform |> adjoint;
end;

# ╔═╡ 78813f7a-5cd8-11eb-2bea-278bd4f36ec0
plot( histogram( rs1Y[3,:], legend = false, yaxis = "raw"
               , title = "Drain Current Distribution" )
    , histogram( us3Y[3,:], legend = false, yaxis = "qt" ) 
    , histogram( us1Y[3,:], legend = false, yaxis = "weighted" )
	, histogram( us2Y[3,:], legend = false, yaxis = "bct" )
    ; layout = (4,1) )
#plot( histogram( us1Y[1,:], legend = false, title = "raw", yaxis = "vth")
#	, histogram( us2Y[1,:], legend = false, title = "bct")
#	, histogram( us3Y[1,:], legend = false, title = "yjt")
#	, histogram( us1Y[2,:], legend = false, yaxis = "vdsat")
#	, histogram( us2Y[2,:], legend = false)
#	, histogram( us3Y[2,:], legend = false)
#	, histogram( us1Y[3,:], legend = false, yaxis = "id")
#	, histogram( usCY[3,:], legend = false)
#	, histogram( us3Y[3,:], legend = false)
#	, histogram( us1Y[4,:], legend = false, yaxis = "gm")
#	, histogram( us2Y[4,:], legend = false)
#	, histogram( us3Y[4,:], legend = false)
#	, histogram( us1Y[5,:], legend = false, yaxis = "gmb")
#	, histogram( us2Y[5,:], legend = false)
#	, histogram( us3Y[5,:], legend = false)
#	, histogram( us1Y[6,:], legend = false, yaxis = "gds")
#	, histogram( us2Y[6,:], legend = false)
#	, histogram( us3Y[6,:], legend = false)
#	, histogram( us1Y[7,:], legend = false, yaxis = "fug")
#	, histogram( us2Y[7,:], legend = false)
#	, histogram( us3Y[7,:], legend = false)
#	, layout = (7,3) )

# ╔═╡ 297b36fc-616b-11eb-37d9-2d761bb2e287
begin
	simData.gmid = simData.gm ./ simData.id;
	simData.idW = simData.id ./ simData.W;
	simData.A0 = simData.gm ./ simData.gds;
	
	maskMatrix = Matrix(isinf.(simData) .| isnan.(simData));
	mask = (vec ∘ collect)(sum(maskMatrix, dims = 2) .== 0 );
	dataFrame = simData[mask, :];
end;

# ╔═╡ 5ba2fb94-5985-11eb-1710-932d14cb2c51
md"""
# Comparison
"""

# ╔═╡ 19e00e34-55b8-11eb-2ecd-3398e288598a
begin	
    ptmn90modelFile = "../model/op-ptmn90-2021-02-19T09:03:05.608/ptmn90.bson"
    ptmn90model     = BSON.load(ptmn90modelFile);
    ptmn90          = ptmn90model[:model];
    ptmn90paramsX   = ptmn90model[:paramsX];
    ptmn90paramsY   = ptmn90model[:paramsY];
    ptmn90numX      = length(ptmn90paramsX);
    ptmn90numY      = length(ptmn90paramsY);
    ptmn90utX       = ptmn90model[:utX];
    ptmn90utY       = ptmn90model[:utY];
    ptmn90maskBCX   = ptmn90model[:maskX];
    ptmn90maskBCY   = ptmn90model[:maskY];
    ptmn90λ         = ptmn90model[:lambda];
    ptmn90devName   = ptmn90model[:name];
    ptmn90devType   = ptmn90model[:type];

    function ptmn90predict(X)
        X[ptmn90maskBCX,:] = bc.(abs.(X[ptmn90maskBCX,:]); λ = ptmn90λ);
        X′ = StatsBase.transform(ptmn90utX, X);
        Y′ = ptmn90(X′);
        Y = StatsBase.reconstruct(ptmn90utY, Y′);
        Y[ptmn90maskBCY,:] = bc′.(Y[ptmn90maskBCY,:]; λ = ptmn90λ);
        return DataFrame(Float64.(Y'), String.(ptmn90paramsY))
    end;
end;

# ╔═╡ 0d74596a-55be-11eb-0dea-ef5d6ab219c6
begin
    W = rand(filter(w -> w > 2.0e-6, unique(dataFrame.W)));
    L = rand(filter(l -> l < 1.0e-6, unique(dataFrame.L)));
    VB = 0.0;

    dat = simData[ ( (simData.W  .== W)
                  .& (simData.Vbs .== VB)
                  .& (simData.L .== L) ) 
                 , ["id", "Vgs", "Vds" ] ];
    len = size(dat) |> first;
    
    vgs = dat.Vgs;
    qvgs = vgs.^2.0;
    VDS = dat.Vds;
    evds = exp.(VDS);
    len = length(vgs);
    vbc = fill(VB, len);
    rvbc = sqrt.(abs.(vbc));
    wc = fill(W, len);
    lc = fill(L, len);

    sweep = [ wc lc vgs  qvgs VDS evds vbc rvbc ]';

    ptmn90y = ptmn90predict(sweep);
    
    idₚₙ = reshape(ptmn90y.id, (Int(sqrt(len)), Int(sqrt(len))));
    idₜₙ = reshape(dat.id, (Int(sqrt(len)), Int(sqrt(len))));   
end;

# ╔═╡ 57ff103c-55bf-11eb-294a-d5ce2b0557c8
begin
	surface(unique(dat.Vgs), unique(dat.Vds), idₜₙ; c = :blues, legend = false);
	surface!( unique(dat.Vgs), unique(dat.Vds), idₚₙ
            ; c = :heat, legend = false
            , xaxis = "Vgs [V]", yaxis = "Vds [V]", zaxis = "Id [A]"
            , title = "PTM 90nm" )
end

# ╔═╡ 9e3dee36-55bc-11eb-355f-2f28faf37480
begin
    slWC = @bind wC Slider( 1.0e-6 : 2.5e-7 : 4.75e-6
                        , default = 1.0e-6, show_value = true );
    slLC = @bind lC Slider( 3.0e-7 : 1.5e-7 : 3e-6
                        , default = 3.0e-7, show_value = true );
    slVBC = @bind vbC Slider( 0.0 : -0.1 : -1.0
                        , default = 0.0, show_value = true );
    
#     md"""
# W = $(slWC)
# L = $(slLC)
# 
# Vbs = $(slVBC)
#     """
end;

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
# ╟─21b30176-598e-11eb-0322-19cbd312896d
# ╠═2d96bb80-5a39-11eb-1e4f-1dd65b73dbd5
# ╠═3f07f502-5cc8-11eb-33ac-83920c977746
# ╠═bc3a4716-5cd0-11eb-1939-91291bfdf530
# ╠═bceea132-5cd7-11eb-14ef-e70cad73f65f
# ╟─78813f7a-5cd8-11eb-2bea-278bd4f36ec0
# ╠═297b36fc-616b-11eb-37d9-2d761bb2e287
# ╟─5ba2fb94-5985-11eb-1710-932d14cb2c51
# ╠═19e00e34-55b8-11eb-2ecd-3398e288598a
# ╠═0d74596a-55be-11eb-0dea-ef5d6ab219c6
# ╠═57ff103c-55bf-11eb-294a-d5ce2b0557c8
# ╠═9e3dee36-55bc-11eb-355f-2f28faf37480
# ╟─c059152c-5a7b-11eb-062c-9f68eca827dc
# ╠═faad964e-5a7b-11eb-30e5-7552e886738c
# ╟─f9a4397e-5a80-11eb-3843-2db5c69de322
