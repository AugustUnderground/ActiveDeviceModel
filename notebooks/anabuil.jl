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

# ╔═╡ c0f94ffa-647c-11eb-0072-231ab368a7d8
using PlutoUI, Plots, StatsBase, JLD2, BSON, DataFrames, Random, Flux, Zygote, Optim, 
      NumericIO, MLDataUtils

# ╔═╡ ce16822a-6ac9-11eb-2b91-47e2eef82d3a
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
	Core.eval(Main, :(using StatsBase))
end

# ╔═╡ 8222632c-6abf-11eb-2620-3197f758829b
md"""
# Setup
"""

# ╔═╡ 42c16682-6480-11eb-3796-8509a0b773eb
plotly();

# ╔═╡ c85ba5c6-6522-11eb-1756-d3c4c23da612
begin
	boxCox(yᵢ; λ = 0.2) = (λ != 0) ? ((yᵢ.^λ) .- 1) ./ λ : log.(yᵢ) ;
	coxBox(y′; λ = 0.2) = (λ != 0) ? exp.( log.((λ .* y′) .+ 1) ./ λ ) .- 1 : exp.(y′) ;
end;

# ╔═╡ 80f1a55c-6ac0-11eb-04dc-633c2ec34d31
begin
	dataDir = "../../data/";
	nmos90file = dataDir * "ptmn90.jld";
	
	nmos90data = jldopen(nmos90file, "r") do file
	    file["database"];
	end;
	
	nmos90data.QVgs = nmos90data.Vgs.^2.0;
	nmos90data.EVds = exp.(nmos90data.Vds);
	nmos90data.Vgs = round.(nmos90data.Vgs, digits = 2);
	nmos90data.Vds = round.(nmos90data.Vds, digits = 2);
	nmos90data.Vbs = round.(nmos90data.Vbs, digits = 2);
	nmos90data.gmid = nmos90data.gm ./ nmos90data.id;
	nmos90data.A0 = nmos90data.gm ./ nmos90data.gds;
	nmos90data.idW = nmos90data.id ./ nmos90data.W;
end;

# ╔═╡ 09c86a9e-6ac8-11eb-08fc-f5ad2f562543
struct Mapping
    model
    paramsX
    paramsY
    utX
    utY
    lambda
end;

# ╔═╡ 25bb1062-6b6d-11eb-04c4-aba45c2aa22f
md"""
## Statistics
"""

# ╔═╡ 30efcf98-6b6d-11eb-1593-f530a9c22605
begin
    numSamples = 666666;

    idxSamples = StatsBase.sample( MersenneTwister(666)
                                 , 1:size(nmos90data, 1)
                                 , StatsBase.pweights(nmos90data.id)
                                 , numSamples
                                 ; replace = false 
                                 , ordered = false );

    sampleData = nmos90data[idxSamples, : ];
end;

# ╔═╡ 13e844bc-6b7a-11eb-2c5b-07b65ed5c75f
plot( histogram(sampleData.W; yaxis = "W")
    , histogram(boxCox.(sampleData.W))
    , histogram(sampleData.L; yaxis = "L")
    , histogram(boxCox.(sampleData.L))
    , histogram(sampleData.Vds; yaxis = "Vds")
    , histogram(log10.(abs.(sampleData.Vds)))
    , histogram(sampleData.Vgs; yaxis = "Vgs")
    , histogram(boxCox.(abs.(sampleData.Vgs)))
    , histogram(sampleData.Vbs; yaxis = "Vbs")
    , histogram(boxCox.(abs.(sampleData.Vbs)))
    ; layout = (5,2), legend = false )

# ╔═╡ afdcae8a-6b6e-11eb-26bf-31ce46fb0231
plot( histogram(sampleData.id; yaxis = "Id")
    , histogram(boxCox.(sampleData.id))
    , histogram(sampleData.gm; yaxis = "gm")
    , histogram(boxCox.(sampleData.gm))
    , histogram(sampleData.vdsat; yaxis = "vdsat")
    , histogram(boxCox.(sampleData.vdsat))
    , histogram(sampleData.vth; yaxis = "vth")
    , histogram(boxCox.(sampleData.vth))
    , histogram(sampleData.gds; yaxis = "gds")
    , histogram(boxCox.(sampleData.gds))
    , histogram(sampleData.gmb; yaxis = "gmb")
    , histogram(boxCox.(sampleData.gmb))
    , histogram(sampleData.fug; yaxis = "fug")
    , histogram(boxCox.(sampleData.fug))
    ; layout = (7,2), legend = false )

# ╔═╡ 1e8c909a-6b72-11eb-30c9-f3063630fb70
plot( vcat([ [ histogram(sampleData[cxx]) 
             ; histogram(abs.(sampleData[cxx])) 
			 ; histogram(boxCox.(abs.(sampleData[cxx]))) ]
             for cxx in filter( (n) -> startswith(n, "c")
                               , names(sampleData)) ]...)...
    ; legend = false, layout = (6,3))


# ╔═╡ 0313bd14-6aca-11eb-2ea8-31bc376c0cb3
md"""
# Funciton Mappings

#### Default

Mapping Terminal Voltages and Sizing to Operating Point Parameters:

$$\begin{bmatrix}
    W \\ L \\ V_{\text{gs}} \\ V_{\text{ds}} \\ V_{\text{bs}} \\
\end{bmatrix}
↦
\begin{bmatrix}
    i_{\text{d}} \\ g_{\text{m}} \\ g_{\text{ds}} \\ 
    v_{\text{d sat}} \\ v_{\text{th}} \\ \vdots
\end{bmatrix}$$

#### As Function of $v_{\text{dsat}}$

$$\begin{bmatrix}
    i_{\text{d}} \\ L \\ v_{\text{d sat}} \\ V_{\text{ds}} \\ V_{\text{bs}} \\
\end{bmatrix}
↦
\begin{bmatrix}
    W \\ V_{\text{gs}} \\ g_{\text{m}} \\ g_{\text{ds}} \\ 
    v_{\text{th}} \\ \vdots
\end{bmatrix}$$

#### As Function of $\frac{g_{\text{m}}}{i_{\text{d}}}$

$$\begin{bmatrix}
    g_{\text{m}} \\ i_{\text{d}} \\ V_{\text{gs}} \\ V_{\text{ds}} \\ V_{\text{bs}} \\
\end{bmatrix}
↦
\begin{bmatrix}
    W \\ L  \\ g_{\text{ds}} \\ 
    v_{\text{d sat}} \\ v_{\text{th}} \\ \vdots
\end{bmatrix}$$


"""

# ╔═╡ b5dc7438-6ac3-11eb-0f40-2f9cd25f8fdd
begin
    prefixNMOS90vdsat = "../model/des-2021-02-08T17:41:45.706/ptmn90";
    modelNMOS90vdsat = BSON.load(prefixNMOS90vdsat * ".bson");
    
    NMOS90vdsat = Mapping( modelNMOS90vdsat[:model]
                         , modelNMOS90vdsat[:paramsX]
                         , modelNMOS90vdsat[:paramsY]
                         , modelNMOS90vdsat[:utX]
                         , modelNMOS90vdsat[:utY]
                         , modelNMOS90vdsat[:lambda] );

    function predictN90vdsat(X)
        X′ = StatsBase.transform(NMOS90vdsat.utX, X)
        Y′ = NMOS90vdsat.model(X′)
        Y = coxBox.( StatsBase.reconstruct(NMOS90vdsat.utY, Y′)
                   ; λ = NMOS90vdsat.lambda)
        return Float64.(Y)
    end;

end;

# ╔═╡ 1dfaed64-6aca-11eb-0e83-3f47c7abfe44
begin
    prefixNMOS90gmid = "../model/des-2021-02-08T16:38:29.407/ptmn90";
    modelNMOS90gmid = BSON.load(prefixNMOS90gmid * ".bson");

    NMOS90gmid = Mapping( modelNMOS90gmid[:model]
                        , modelNMOS90gmid[:paramsX]
                        , modelNMOS90gmid[:paramsY]
                        , modelNMOS90gmid[:utX]
                        , modelNMOS90gmid[:utY]
                        , modelNMOS90gmid[:lambda] );

    function predictN90gmid(X)
        X′ = StatsBase.transform(NMOS90gmid.utX, X)
        Y′ = NMOS90vdsat.model(X′)
        Y = coxBox.( StatsBase.reconstruct(NMOS90gmid.utY, Y′)
                   ; λ = NMOS90gmid.lambda)
        return Float64.(Y)
    end;

end;

# ╔═╡ bc05bc7e-6ad3-11eb-1afe-0bfcd33073af
md"""
### Comparison / Evaluation
"""

# ╔═╡ 1f248c18-6ad4-11eb-389e-e9bf2198cfd8
traceT = sort( nmos90data[ ( (nmos90data.L .== 500e-9)
                         .& (nmos90data.W .== 2.0e-6)
                         .& (nmos90data.Vds .== 0.6) )
                        , ["vdsat", "idW"] ]
             , "idW" );

# ╔═╡ c7f82f3e-6ad4-11eb-2204-01b12c037b71
begin
	slVds01 = @bind vds01 Slider( 0.01 : 0.01 : 1.20
							    , default = 0.6, show_value = true );
	slId01 = @bind id01 Slider( 1.0e-6 : 2.5e-7 : 50.0e-6
						      , default = 1.0e-6, show_value = true );
	slL01 = @bind l01 Slider( 1.5e-7 : 1.0e-7 : 1.5e-6
						    , default = 3.0e-7, show_value = true );
	
md"""
`Vds` = $(slVds01) V

`Id` = $(slId01) A

`L` = $(slL01) m
"""
end

# ╔═╡ 99e31a6e-6ad4-11eb-05ff-1149f92854bb
begin
    vdsat = 0.05:0.01:0.5;
    len = length(traceT.vdsat);

    x = [ fill(l01, len)'
        ; fill(id01, len)'
        ; traceT.vdsat' #collect(vdsat)'
        ; fill(vds01, len)'
        ; exp.(fill(vds01, len))' ]

    y = predictN90vdsat(x);

    traceP = sort( DataFrame(vdsat = traceT.vdsat, idW = y[1,:])
                 , "vdsat" )
end;

# ╔═╡ 5fb3cafa-6ad4-11eb-14de-996dc1e904cf
begin
    plot( traceT.vdsat, traceT.idW
        ; lab = "Truth @ L = 250nm, Vds = 0.6V"
		, yscale = :log10, w = 2
        , xaxis = "vdsat", yaxis = "id/W" );
    plot!( traceP.vdsat, traceP.idW
         ; lab = "Apprx @ L = $(l01)m, Vds = $(vds01)V"
         , legend = :bottomright, w = 2 )
end

# ╔═╡ aba0fc00-6557-11eb-2788-7508e5ae831c
md"""
# Analog Building Blocks

## Current Mirror

Function mapping for Building Block:

$$\begin{bmatrix}
I_{\text{bias}} \\ 
M = \frac{M_{\text{cm12}}}{M_{\text{cm11}}} \\ 
I_{\text{out}} = I_{\text{d, M1}} \approx M \cdot I_{\text{bias}} \\
\frac{g_{\text{m,M0}}}{I_{\text{d,M0}}} \,\|\,v_{\text{dsat,M0}} \\ 
f_{\text{ug, M0}} \,\|\, \sigma
\end{bmatrix}
\mapsto
\begin{bmatrix}
W_{\text{in}} = M_{\text{cm11}} \cdot W \\ 
W_{\text{out}} = M_{\text{cm12}} \cdot W \\ 
L \\ 
V_{\text{in}} = V_{\text{ds, M0}} \\
V_{\text{out}} = V_{\text{ds, M1}}
\end{bmatrix}$$
"""

# ╔═╡ 3d9fe546-3e12-11eb-3e0d-7f5e9d423e92
md"""
## Differential Pair (DP)

Proposed function mapping for Building Block:

"""

# ╔═╡ Cell order:
# ╟─8222632c-6abf-11eb-2620-3197f758829b
# ╠═c0f94ffa-647c-11eb-0072-231ab368a7d8
# ╠═ce16822a-6ac9-11eb-2b91-47e2eef82d3a
# ╠═42c16682-6480-11eb-3796-8509a0b773eb
# ╠═c85ba5c6-6522-11eb-1756-d3c4c23da612
# ╠═80f1a55c-6ac0-11eb-04dc-633c2ec34d31
# ╠═09c86a9e-6ac8-11eb-08fc-f5ad2f562543
# ╟─25bb1062-6b6d-11eb-04c4-aba45c2aa22f
# ╠═30efcf98-6b6d-11eb-1593-f530a9c22605
# ╠═13e844bc-6b7a-11eb-2c5b-07b65ed5c75f
# ╠═afdcae8a-6b6e-11eb-26bf-31ce46fb0231
# ╠═1e8c909a-6b72-11eb-30c9-f3063630fb70
# ╟─0313bd14-6aca-11eb-2ea8-31bc376c0cb3
# ╠═b5dc7438-6ac3-11eb-0f40-2f9cd25f8fdd
# ╠═1dfaed64-6aca-11eb-0e83-3f47c7abfe44
# ╟─bc05bc7e-6ad3-11eb-1afe-0bfcd33073af
# ╠═1f248c18-6ad4-11eb-389e-e9bf2198cfd8
# ╠═99e31a6e-6ad4-11eb-05ff-1149f92854bb
# ╟─5fb3cafa-6ad4-11eb-14de-996dc1e904cf
# ╟─c7f82f3e-6ad4-11eb-2204-01b12c037b71
# ╟─aba0fc00-6557-11eb-2788-7508e5ae831c
# ╟─3d9fe546-3e12-11eb-3e0d-7f5e9d423e92
