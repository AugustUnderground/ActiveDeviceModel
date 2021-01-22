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
using DataFrames, StatsBase, JLD2, StatsPlots, PlutoUI, DataInterpolations, PyCall, ScikitLearn, Optim, Random, Statistics, Distributions, BSON, Flux, Zygote, CUDA, PyCall, ScikitLearn, NNlib, CSVFiles, Lazy, BoxCoxTrans, YeoJohnsonTrans

# ╔═╡ 31c636ac-55b8-11eb-19d6-8dc9af976a24
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

# ╔═╡ 0f54b05a-54ff-11eb-21e4-511378903bfe
md"""
# NMOS

## gm / id (Data Base)
"""

# ╔═╡ 5b2ea554-5a3f-11eb-3f75-2d7f8ce5a368
begin
	@sk_import preprocessing: PowerTransformer;
	@sk_import preprocessing: QuantileTransformer;
end

# ╔═╡ 478b1cde-3e34-11eb-367b-476c0408e6c3
joblib = pyimport("joblib");

# ╔═╡ 99ba2bec-4034-11eb-045f-49b2e8eca1de
plotly();

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

# ╔═╡ d5132790-5ccc-11eb-21f6-810148865c86


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
	
	coxX = hcat([ bc(rX; λ = 0.2) for rX in eachrow(ur1X)]...)';
	coxY = hcat([ bc(rY; λ = 0.2) for rY in eachrow(ur1Y)]...)';
	
	ut2X = StatsBase.fit(UnitRangeTransform, coxX; dims = 2, unit = true); 
	ut2Y = StatsBase.fit(UnitRangeTransform, coxY; dims = 2, unit = true);
end;

# ╔═╡ bceea132-5cd7-11eb-14ef-e70cad73f65f
begin
	us1X = StatsBase.transform(ut1X, Matrix(simSample[:,paramsX])');
	us1Y = StatsBase.transform(ut1Y, Matrix(simSample[:,paramsY])');
	
	usCX = hcat([ bc(rX; λ = 0.2) for rX in eachrow(us1X)]...)';
	usCY = hcat([ bc(rY; λ = 0.2) for rY in eachrow(us1Y)]...)';
	
	us2X = StatsBase.transform(ut2X, usCX);
	us2Y = StatsBase.transform(ut2Y, usCY);
end

# ╔═╡ 78813f7a-5cd8-11eb-2bea-278bd4f36ec0
histogram(us2Y[3,:])

# ╔═╡ 7b5eefa8-5cda-11eb-1427-e75d848c7c54
begin
	u1XT = StatsBase.transform(ut1X, bc_xt);
	cxXT = hcat([ bc(rX; λ = 0.2) for rX in eachrow(u1XT)]...)';
	u2XT = StatsBase.transform(ut2X, cxXT);
end

# ╔═╡ fda2ce9e-5c82-11eb-2505-c7311cf10ed0
begin
	rawId = simData.id;
	urtId = StatsBase.fit(UnitRangeTransform, rawId);
	urId = StatsBase.transform(urtId, rawId); # .+ 1;
	λ = 0.2; #BoxCoxTrans.lambda(weiSampled.id).value;
	brId = abs.(bc(rawId; λ = λ));
	bId = (bc(urId; λ = λ));
	#bId = abs.(bc(weiSampled.id; λ = λ)); 
	#bId′ = bc′(bId, λ = λ);
end;

# ╔═╡ e56185f4-5a3d-11eb-1337-57b0f110e054
plot( histogram(rawId; title = "id", yaxis = "Raw")
	, histogram(urId; yaxis = "Unit")
	, histogram(brId; yaxis = "Raw Cox")
	, histogram(bId; yaxis = "Unit Cox")
	; layout = (4,1)
	, legend = false )

# ╔═╡ 7d31084a-5a58-11eb-3565-a96777aca557
md"""
#### Multiple Sub Samples

Iteratively sample half the population. 

Model **doesn't** learn from the resulting data set 🐣
"""

# ╔═╡ bf169d88-5cbf-11eb-3cf8-6fb960492987
begin
	allSamples = size(simData)[1];
	minSamples = 666666;
	numSamples = @>> Lazy.range() map(x -> (allSamples / (2^x)));
	numSmpl = Int.(ceil.(takewhile((x) -> x > minSamples, numSamples)));
	typSmpl = String.(take(length(numSmpl), Lazy.cycle(["id", "gm"])));
	
	subSampled = reduce( (dat, smp) -> dat[ StatsBase.sample( MersenneTwister(666)
                              			   					, 1:(dat |> size |> first)
					   					   					, pweights(dat[:,smp[2]])
															, smp[1]
                              			   					; replace = false )
										  , : ]
		  			   , zip(numSmpl, typSmpl); init = simData );
		
	weiSampled = simData[ StatsBase.sample( MersenneTwister(666)
                              			  , 1:(simData |> size |> first)
										  , pweights(simData.id)
                             			  , Lazy.last(numSmpl)
                            			  ; replace = false )
						, : ];
	
	regSampled = simData[ StatsBase.sample( MersenneTwister(666)
                              			  , 1:(simData |> size |> first)
                              			  , Lazy.last(numSmpl)
                              			  ; replace = false ) 
					    , : ];
end;

# ╔═╡ 9e85199c-5c00-11eb-1e91-97a23967fd7d
md"""
For $(first(size(subSampled))) Samples
"""

# ╔═╡ 27cd1128-5c06-11eb-2f2f-2bbe580566f7
md"""
### Scikit Transformers
"""

# ╔═╡ 8539fb7a-5c0e-11eb-0775-d98cd6cd5437
id = reshape(weiSampled.id, (size(weiSampled.id)..., 1));

# ╔═╡ 04a38b9a-5c07-11eb-0348-55b935cb268c
begin
	qtu = QuantileTransformer( output_distribution = "uniform"
                             , random_state = 666 );
	qIdu = qtu.fit_transform(reshape(weiSampled.id, (size(weiSampled.id)..., 1)));
end;

# ╔═╡ ce78d548-5c0b-11eb-3944-0945cac84c22
begin
	qtn = QuantileTransformer( output_distribution = "normal"
                             , random_state = 666 );
	qIdn = qtn.fit_transform(reshape(weiSampled.id, (size(weiSampled.id)..., 1)));
end;

# ╔═╡ 15b3073a-5c07-11eb-0691-27087cb22f70
begin
	pt = PowerTransformer(method = "box-cox");
	pIdt = pt.fit_transform(reshape(weiSampled.id, (size(weiSampled.id)..., 1)));
	pId = reshape(pIdt, length(pIdt))
end;

# ╔═╡ a82d6e12-5c0b-11eb-13d7-c7507911d7f9
begin
	histogram( [weiSampled.id pId qIdu qIdn]
			 ; alpha = 0.5)
end

# ╔═╡ 2fa2e3a6-5c91-11eb-2a4c-619062a93698
md"""
### Comparing Cox
🐍 🔫 🐔
"""

# ╔═╡ 634da1aa-5c91-11eb-0142-e95e6005b872
begin	
	myBCTur = StatsBase.fit(UnitRangeTransform, bId; unit = true);
	myBCT = StatsBase.transform(myBCTur, bId);
	pyBCTur = StatsBase.fit(UnitRangeTransform, pId; unit = true);
	pyBCT = StatsBase.transform(pyBCTur, pId);
end;

# ╔═╡ 07a01308-5c91-11eb-1816-75a97d54eb57
histogram( [myBCT pyBCT]; alpha = 0.5)

# ╔═╡ 5ba2fb94-5985-11eb-1710-932d14cb2c51
md"""
# Comparison
"""

# ╔═╡ 19e00e34-55b8-11eb-2ecd-3398e288598a
begin	
	modelPathₙ = "../model/ptmn90-2021-01-21T15:04:33.212/ptmn90";
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
	
	idₚ = reshape( yₙ[first(indexin(["id"], paramsYₙ)), :]
				 , (Int(sqrt(len)), Int(sqrt(len))));
	idₜ = reshape( dat.id
				 , (Int(sqrt(len)), Int(sqrt(len))));	
end;

# ╔═╡ 57ff103c-55bf-11eb-294a-d5ce2b0557c8
begin
	surface(unique(dat.Vgs), unique(dat.Vds), idₜ; c = :blues);
	surface!(unique(dat.Vgs), unique(dat.Vds), idₚ; c = :reds)
end

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

# ╔═╡ 9345c24e-5b23-11eb-03ff-11332c241f4e
sort!(simData, [:Vgs, :Vds, :W])

# ╔═╡ b8c2b8a8-5b21-11eb-1621-db196dd4947f
begin
	binsVgs = unique(simData.Vgs);
	binsVds = unique(simData.Vds);
	binsW   = unique(simData.W);
	binsL   = unique(simData.L);
end;

# ╔═╡ 1abf033e-5b24-11eb-245c-834f05f57066
diff(simData.L)

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
# ╠═21b30176-598e-11eb-0322-19cbd312896d
# ╠═d5132790-5ccc-11eb-21f6-810148865c86
# ╠═2d96bb80-5a39-11eb-1e4f-1dd65b73dbd5
# ╠═fda2ce9e-5c82-11eb-2505-c7311cf10ed0
# ╠═e56185f4-5a3d-11eb-1337-57b0f110e054
# ╟─7d31084a-5a58-11eb-3565-a96777aca557
# ╠═bf169d88-5cbf-11eb-3cf8-6fb960492987
# ╟─9e85199c-5c00-11eb-1e91-97a23967fd7d
# ╟─27cd1128-5c06-11eb-2f2f-2bbe580566f7
# ╠═8539fb7a-5c0e-11eb-0775-d98cd6cd5437
# ╠═04a38b9a-5c07-11eb-0348-55b935cb268c
# ╠═ce78d548-5c0b-11eb-3944-0945cac84c22
# ╠═15b3073a-5c07-11eb-0691-27087cb22f70
# ╟─a82d6e12-5c0b-11eb-13d7-c7507911d7f9
# ╟─2fa2e3a6-5c91-11eb-2a4c-619062a93698
# ╠═634da1aa-5c91-11eb-0142-e95e6005b872
# ╟─07a01308-5c91-11eb-1816-75a97d54eb57
# ╟─5ba2fb94-5985-11eb-1710-932d14cb2c51
# ╠═19e00e34-55b8-11eb-2ecd-3398e288598a
# ╠═917a3ff2-55bb-11eb-36f3-fb1d62827973
# ╠═0d74596a-55be-11eb-0dea-ef5d6ab219c6
# ╟─57ff103c-55bf-11eb-294a-d5ce2b0557c8
# ╟─9e3dee36-55bc-11eb-355f-2f28faf37480
# ╟─c059152c-5a7b-11eb-062c-9f68eca827dc
# ╠═faad964e-5a7b-11eb-30e5-7552e886738c
# ╟─f9a4397e-5a80-11eb-3843-2db5c69de322
# ╠═9345c24e-5b23-11eb-03ff-11332c241f4e
# ╠═b8c2b8a8-5b21-11eb-1621-db196dd4947f
# ╠═1abf033e-5b24-11eb-245c-834f05f57066
