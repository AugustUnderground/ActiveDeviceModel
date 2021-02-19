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

# ╔═╡ aa866870-50f9-11eb-268e-8b614dd7f83c
using Plots, PlutoUI, Flux, Zygote, CUDA, BSON, JLD2,
      StatsBase, NNlib, FiniteDifferences, DataFrames,
      NumericIO, LaTeXStrings, Chain, ForwardDiff

# ╔═╡ c7e55070-50f9-11eb-29fa-fd2d4ce6db29
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
	Core.eval(Main, :(using StatsBase))
end

# ╔═╡ 69d38de8-50fa-11eb-1e3d-4bc49627f622
md"""
# Machine Learnign Model Evaluation

## Imports
"""

# ╔═╡ da09d154-50f9-11eb-3ee8-9112221e4658
#pyplot();
plotly();
#inspectdr();

# ╔═╡ 3950a4ca-6b90-11eb-3d01-99f0fa677fe4
struct MOSFET
    model
    paramsX
    paramsY
    utX
    utY
    maskX
    maskY
    lambda
end;

# ╔═╡ 39cfd4de-72ca-11eb-1b25-f37f84953389
struct OPERATOR
    parameter
    model
    paramsX
    paramsY
    utX
    utY
    maskX
    maskY
    lambda
end;

# ╔═╡ ad021952-6b91-11eb-03b5-4daf591847cd
begin
	boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
	coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);
end;

# ╔═╡ a12acdfe-50ff-11eb-018f-df481ecc63df
md"""
## Loading Models
"""

# ╔═╡ e051a73a-50f9-11eb-3a7a-11906e8f508d
begin	
	nmos90path = "../model/op-ptmn90-2021-02-19T09:03:05.608/ptmn90.bson"
	nmos90file = BSON.load(nmos90path);

    ptmn90 = MOSFET( nmos90file[:model]
                   , String.(nmos90file[:paramsX])
                   , String.(nmos90file[:paramsY])
                   , nmos90file[:utX]
                   , nmos90file[:utY]
                   , nmos90file[:maskX]
                   , nmos90file[:maskY]
                   , nmos90file[:lambda] );

    function nmos90(Vgs, Vds, Vbs, W, L)
        input = adjoint(hcat(W, L, Vgs, Vgs.^2.0, Vds, exp.(Vds), Vbs, sqrt.(abs.(Vbs))));
        #input = DataFrame(valuesX, ptmn90.paramsX);
        X = Matrix(input);
        X[ptmn90.maskX,:] = boxCox.(abs.(X[ptmn90.maskX,:]); λ = ptmn90.lambda);
        X′ = StatsBase.transform(ptmn90.utX, X);
        Y′ = ptmn90.model(X′);
        Y = StatsBase.reconstruct(ptmn90.utY, Y′);
        Y[ptmn90.maskY,:] = coxBox.(Y[ptmn90.maskY,:]; λ = ptmn90.lambda);
        return DataFrame(Float64.(Y'), String.(ptmn90.paramsY))
    end;

end;

# ╔═╡ 1d30714c-50fa-11eb-30fc-c9b7e9ad1017
begin	
	pmos90path = "../model/op-ptmp90-2021-02-19T09:21:58.867/ptmp90.bson"
	pmos90file = BSON.load(pmos90path);

    ptmp90 = MOSFET( pmos90file[:model]
                   , String.(pmos90file[:paramsX])
                   , String.(pmos90file[:paramsY])
                   , pmos90file[:utX]
                   , pmos90file[:utY]
                   , pmos90file[:maskX]
                   , pmos90file[:maskY]
                   , pmos90file[:lambda] );

    function pmos90(X)
        X′ = StatsBase.transform(ptmp90.utX, X)
        YZ′ = ptmp90.model(X′)
        Y′ = YZ′[1:length(ptmp90.paramsY), :];
        Z′ = YZ′[(length(ptmp90.paramsY) + 1):end, :];
        Y = coxBox.(StatsBase.reconstruct(ptmp90.utY, Y′); λ = ptmp90.lambda );
        Z = coxBox.(StatsBase.reconstruct(ptmp90.utZ, Z′); λ = ptmp90.lambda );
        return DataFrame(Float64.([Y;Z])', [ptmp90.paramsY ; ptmp90.paramsZ]')
    end;

end;

# ╔═╡ 0ababd1a-72ca-11eb-393b-ef712c74190c
begin
    vdsatFile = BSON.load("../model/vdsat-ptmn45-2021-02-19T14:44:39.674/ptmn45.bson")

    ptmn45vdsat = OPERATOR( vdsatFile[:parameter]
                          , vdsatFile[:model]
                          , vdsatFile[:paramsX]
                          , vdsatFile[:paramsY]
                          , vdsatFile[:utX]
                          , vdsatFile[:utY]
                          , vdsatFile[:maskX]
                          , vdsatFile[:maskY]
                          , vdsatFile[:lambda] );

    function nmosvdsat(X)
        X[ptmn45vdsat.maskX,:] = boxCox.( abs.(X[ptmn45vdsat.maskX,:])
                                          ; λ = ptmn45vdsat.lambda);
        X′ = StatsBase.transform(ptmn45vdsat.utX, X);
        Y′ = ptmn45vdsat.model(X′);
        Y = StatsBase.reconstruct(ptmn45vdsat.utY, Y′);
        Y[ptmn45vdsat.maskY,:] = coxBox.( Y[ptmn45vdsat.maskY,:]
                                          ; λ = ptmn45vdsat.lambda);
        return DataFrame(Float64.(Y'), String.(ptmn45vdsat.paramsY))
    end;
end;

# ╔═╡ 7d0501a4-54c3-11eb-3e5c-275701e033ea
md"""
## Exploring the Device Models
"""

# ╔═╡ f7d15572-50f9-11eb-2923-fbb666503e9d
begin
    sliderVB = @bind VB Slider( -1.20 : 0.01 : 0.00
                              , default = 0.0, show_value = false );
    sliderW = @bind W Slider( 1.0e-6 : 0.5e-7 : 100.0e-6
                           , default = 3.0e-6, show_value = false );
    sliderL = @bind L Slider( 2e-7 : 1.0e-7 : 20.0e-6
                           , default = 3.0e-7, show_value = false );

md"""
`W` = $(sliderW) $(formatted(W, :ENG, ndigits = 2)) m 
`L` = $(sliderL) $(formatted(L, :ENG, ndigits = 2)) m

`Vbs` = $(sliderVB) $(formatted(VB, :ENG, ndigits = 2)) V
"""
end

# ╔═╡ e702caf0-720f-11eb-2780-1770c60d8247
begin
    #L = rand(1e-7:1e-5);
    #W = rand(1e-6:1e-4);
    VGD = hcat(vec([[i,j] for i = 0.0:0.01:1.2, j = 0.0:0.01:1.2])...);
    VGS = VGD[2,:];
    VDS = VGD[1,:];
    LEN = length(VGS);
    VBS = fill(VB, LEN);
    WS = fill(W, LEN);
    LS = fill(L, LEN);

    nmosOPt = nmos90(VGS, VDS, VBS, WS, LS);

    nmos90id = reshape( nmosOPt.id, (Int(sqrt(LEN)), Int(sqrt(LEN))));	

    surface( unique(VGS), unique(VDS), nmos90id
           ; c = :blues, legend = false
           , xaxis = "Vgs [V]", yaxis = "Vds [V]", zaxis = "Id [A]" 
           , title = "Characteristic" )
end

# ╔═╡ e1d7d448-72c9-11eb-3ea6-99c8336551de
md"""
## Exploring the Design Models
"""

# ╔═╡ 727798a8-72de-11eb-03ff-6920fde5836d
begin
    f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
    xf = rand(5)
    g = x -> ForwardDiff.gradient(f, x);
    g(xf)
end

# ╔═╡ 46da2f52-5b1c-11eb-3476-73665e0f2366
md"""
### Numeric Analysis
"""

# ╔═╡ aa26b788-72cb-11eb-3508-05c306d752d2
begin
    sliderLL = @bind LL Slider( 1e-7 : 1e-7 : 1e-6
                              , default = 3e-7, show_value = false );
    sliderId = @bind Id Slider( 1e-6 : 1e-6 : 100e-6
                              , default = 25e-6, show_value = false );
    sliderVD = @bind VD Slider( 0.00 : 0.01 : 1.20
                              , default = 0.6, show_value = false );
md"""
`L` = $(sliderLL) $(formatted(LL, :ENG, ndigits = 2)) m
`Id` = $(sliderId) $(formatted(Id, :ENG, ndigits = 2)) m

`Vds` = $(sliderVD) $(formatted(VD, :ENG, ndigits = 2)) V

"""
end

# ╔═╡ 2f5dfe0c-72d1-11eb-30d0-a72317580b81
begin
    function Jd(vdsat)
        res = nmosvdsat([LL Id vdsat VD exp(VD) VB sqrt(abs(VB))]');
        return res.Jd
    end;

    function ∇Jd(vdsat)
        return central_fdm(5,1)(Jd, vdsat)[1]
    end;
end;

# ╔═╡ 59f01d36-72cb-11eb-34dd-017c0b53426a
begin
    len = 100;
    vdsat = range(0.05, stop = 0.5, length = len);

    l = fill(LL, len);
    id = fill(Id, len);
    vds = fill(VD, len);
    evds = exp.(vds);
    vbs = fill(VB, len);
    rvbs = sqrt.(abs.(vbs));

    x = [ l id vdsat vds evds vbs rvbs ]';
    suggestion = nmosvdsat(x);
    nmosJd = suggestion.Jd;
    ∂Jd_∂vdsat = ForwardDiff.gradient.(Jd, vdsat);
    #∂Jd_∂vdsat = ∇Jd.(vdsat);
end;

# ╔═╡ fee94b78-72cd-11eb-049f-81d62d5f6c92
begin
    plot(vdsat, suggestion.Jd; yscale = :log10
        , xaxis = "vdsat", yaxis = "Jd", w = 2 );
    plot!( vdsat[2:end], diff(suggestion.Jd) ./ diff(vdsat); w = 2 )
end

# ╔═╡ 8e79cbc0-5bdc-11eb-06a5-29dcd6b17399
function drainCurrent(vgs, vds, w, l)
	pY = predictₙ( [ vgs'  ; (vgs.^2)'
				   ; vds' ; (ℯ.^vds)'
				   ; w' ; l' ] );
	return pY[first(indexin(["id"], paramsYₙ)), :][1];
end;

# ╔═╡ 843a61ec-54e3-11eb-0029-fbfd2976f29b
md"""
## Exploring Design Model

$$\begin{bmatrix}
L \\ i_{\text{d}} \\ \frac{g_{\text{m}}}{i_{\text{d}}} \\ V_{\text{ds}} \\
\end{bmatrix}
\mapsto
\begin{bmatrix}
V_{\text{gs}} \\ \frac{i_{\text{d}}}{W} \\ A_{0} \\ f_{\text{ug}} \\
\end{bmatrix}$$
"""

# ╔═╡ 0b4ab9d6-5661-11eb-1c04-21ea0f3916e1
begin
	idWplot = plot( gmid, pᵧ[2,:]
				  ; yscale = :log10
				  , yaxis = "id/W [A/m]"
				  , xaxis = "gm/id [S/A]"
				  , title = "Current Density over gm/id"
				  , legend = false );
	A0plot = plot( gmid, pᵧ[3,:]
				  ; yscale = :log10
				  , yaxis = "A0"
				  , xaxis = "gm/id [S/A]"
				  , title = "Self Gain over gm/id"
				  , legend = false );
	plot( idWplot, A0plot, layout = (1,2));
end

# ╔═╡ 0e78eab8-54e4-11eb-07e5-0f5b4e63d5e8
begin
	sliderVdsᵧ = @bind Vdsᵧ Slider( 0.01 : 0.01 : 1.20
								  , default = 0.6, show_value = true );
	sliderIdᵧ = @bind Idᵧ Slider( 10e-6 : 1.0e-7 : 75e-6
						        , default = 25e-6, show_value = true );
	sliderLᵧ = @bind Lᵧ Slider( 3e-7 : 1.0e-7 : 1.5e-6
							  , default = 3e-7, show_value = true );
	sliderGᵧ = @bind gmidᵧ Slider( 1 : 0.5 : 25
							  , default = 10, show_value = true );
	md"""
	`L` = $(sliderLᵧ) m
	
	`Id` = $(sliderIdᵧ) A
	
	`gm/id` = $(sliderGᵧ) S/A
	
	`Vds` =$(sliderVdsᵧ) V
	"""
end

# ╔═╡ 8cb6ff56-54e3-11eb-0c02-713c6d2be8b5
begin
	#gmid = 1.0:0.25:25.0;
	#len  = length(gmid);
	
	#xᵧ = [ fill(Lᵧ, 1, len)
	#	 ; fill(Idᵧ, 1, len)
	#	 ; collect(gmid)'
	#	 ; (gmid .* Idᵧ)'
	#	 ; fill(Vdsᵧ, 1, len)
	#	 ; exp.(fill(Vdsᵧ, 1, len)) ];
	#pᵧ = predictᵧ(xᵧ);	
end

# ╔═╡ 4702177c-5658-11eb-1912-6b7b95c8f221
#begin
#	∇gmid = (x) -> grad(central_fdm(5,1), (x) -> predictᵧ(x)[2,1], x) |> first;
#	∂gmid = hcat(map(∇gmid, eachcol(xᵧ))...)[1,:];
#	∇gmid(xᵧ[:,1])
#end

# ╔═╡ 3dbf3974-54fa-11eb-1901-c310b0dd8685
md"""
### Optimization

$(PlutoUI.LocalResource("./symamp.png"))

#### Find Operating Point

1. Define KCL-Error-Function
1. Must contain all Transistors
1. Add Currents add nodes
1. Still done manually

Known Volatges:

$$\phi_{dd} = 1.2\,\text{V}$$
$$\phi_{ss} = 0.0\,\text{V}$$
$$\phi_{I1} = 0.7\,\text{V}$$
$$\phi_{I2} = 0.7\,\text{V}$$
$$\phi_{bias} = 0.8\,\text{V}$$

Find:

$$\phi_{0} = \begin{bmatrix}
\phi_{x} \\
\phi_{u} \\
\phi_{w} \\
\phi_{c} \\
\phi_{b} \\
\phi_{o} \\
\end{bmatrix}$$

So that KCL checks out.

#### Find W/L for all transistors

1. Based on **Find Operating Point**
1. Define Cost-Function with MNA
1. Optimize W/L

"""

# ╔═╡ b6616c00-55a0-11eb-346b-99831a762e03
begin
	ϕDD = 1.2;
	ϕSS = 0.0;
	ϕI1 = 0.7;
	ϕI2 = 0.7;

	ib = 50e-7;
	
	W₁₂ = 2e-6;
	W₃₄ = 4e-6;
	W₅₆ = 4e-6;
	W₇₈ = 3e-6;
	W₉₀ = 3e-6;
	
	Lₘᵢₙ = 3e-7;
end;

# ╔═╡ 529afb34-55a0-11eb-36e7-45fdb7453178
function kcl(ϕ)
    #ϕx, ϕu, ϕw, ϕc, ϕo, ϕB = ϕ;
    id1 , id2 , id3 , id4 , id5 , id6 , id7 , id8 , id9 , id0 = first.(
        [ nmos((ϕI1 - ϕ[1]), (ϕ[2] - ϕ[1]), W₁₂, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos((ϕI2 - ϕ[1]), (ϕ[3] - ϕ[1]), W₁₂, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , pmos((ϕDD - ϕ[2]), (ϕDD - ϕ[2]), W₃₄, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , pmos((ϕDD - ϕ[2]), (ϕDD - ϕ[4]), W₃₄, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , pmos((ϕDD - ϕ[3]), (ϕDD - ϕ[3]), W₅₆, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , pmos((ϕDD - ϕ[3]), (ϕDD - ϕ[5]), W₅₆, Lₘᵢₙ)[indexin(["id"], paramsYₚ),1]
        , nmos(ϕ[6], ϕ[6], W₇₈, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos(ϕ[6], ϕ[1], W₇₈, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos(ϕ[4], ϕ[4], W₉₀, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1]
        , nmos(ϕ[4], ϕ[5], W₉₀, Lₘᵢₙ)[indexin(["id"], paramsYₙ),1] ]);
    ΔϕDD = abs((id4 + id3) - (id5 + id6));
    Δϕu  = abs(id3 - id1);
    Δϕw  = abs(id5 - id2);
    Δϕx  = abs((id1 - id2) - id8);
    Δϕc  = abs(id4 - id9);
    ΔϕB  = abs(id7 - ib);
    Δϕo  = abs(id5 - id0);
    return [ ΔϕDD, Δϕu , Δϕw , Δϕx , Δϕc , ΔϕB , Δϕo ];
end;

# ╔═╡ 29ab0e1e-559e-11eb-2d50-cbfd0e603acb
function cost()
	return (1 / abs(sum(kcl(x))))
end;

# ╔═╡ e634bc18-67c0-11eb-31fd-efe7f3747e4b


# ╔═╡ Cell order:
# ╟─69d38de8-50fa-11eb-1e3d-4bc49627f622
# ╠═aa866870-50f9-11eb-268e-8b614dd7f83c
# ╠═c7e55070-50f9-11eb-29fa-fd2d4ce6db29
# ╠═da09d154-50f9-11eb-3ee8-9112221e4658
# ╠═3950a4ca-6b90-11eb-3d01-99f0fa677fe4
# ╠═39cfd4de-72ca-11eb-1b25-f37f84953389
# ╠═ad021952-6b91-11eb-03b5-4daf591847cd
# ╟─a12acdfe-50ff-11eb-018f-df481ecc63df
# ╠═e051a73a-50f9-11eb-3a7a-11906e8f508d
# ╠═1d30714c-50fa-11eb-30fc-c9b7e9ad1017
# ╠═0ababd1a-72ca-11eb-393b-ef712c74190c
# ╟─7d0501a4-54c3-11eb-3e5c-275701e033ea
# ╟─e702caf0-720f-11eb-2780-1770c60d8247
# ╟─f7d15572-50f9-11eb-2923-fbb666503e9d
# ╟─e1d7d448-72c9-11eb-3ea6-99c8336551de
# ╠═2f5dfe0c-72d1-11eb-30d0-a72317580b81
# ╠═727798a8-72de-11eb-03ff-6920fde5836d
# ╠═59f01d36-72cb-11eb-34dd-017c0b53426a
# ╠═46da2f52-5b1c-11eb-3476-73665e0f2366
# ╠═fee94b78-72cd-11eb-049f-81d62d5f6c92
# ╠═aa26b788-72cb-11eb-3508-05c306d752d2
# ╠═8e79cbc0-5bdc-11eb-06a5-29dcd6b17399
# ╟─843a61ec-54e3-11eb-0029-fbfd2976f29b
# ╠═0b4ab9d6-5661-11eb-1c04-21ea0f3916e1
# ╟─0e78eab8-54e4-11eb-07e5-0f5b4e63d5e8
# ╠═8cb6ff56-54e3-11eb-0c02-713c6d2be8b5
# ╠═4702177c-5658-11eb-1912-6b7b95c8f221
# ╟─3dbf3974-54fa-11eb-1901-c310b0dd8685
# ╠═b6616c00-55a0-11eb-346b-99831a762e03
# ╠═529afb34-55a0-11eb-36e7-45fdb7453178
# ╠═29ab0e1e-559e-11eb-2d50-cbfd0e603acb
# ╠═e634bc18-67c0-11eb-31fd-efe7f3747e4b
