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
using Plots, PlutoUI, Flux, Zygote, CUDA, BSON, 
      StatsBase, NNlib, FiniteDifferences, DataFrames

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
    paramsZ
    utX
    utY
    utZ
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
	nmos90path = "../model/ptmn90-2021-02-11T17:52:16.141/ptmn90.bson"
	nmos90file = BSON.load(nmos90path);

    ptmn90 = MOSFET( nmos90file[:model]
                   , String.(nmos90file[:paramsX])
                   , String.(nmos90file[:paramsY])
                   , String.(nmos90file[:paramsZ])
                   , nmos90file[:utX]
                   , nmos90file[:utY]
                   , nmos90file[:utZ]
                   , nmos90file[:lambda] );

    function nmos90(X)
        X′ = StatsBase.transform(ptmn90.utX, X)
        YZ′ = ptmn90.model(X′)
        Y′ = YZ′[1:length(ptmn90.paramsY), :];
        Z′ = YZ′[(length(ptmn90.paramsY) + 1):end, :];
        Y = coxBox.(StatsBase.reconstruct(ptmn90.utY, Y′); λ = ptmn90.lambda );
        Z = coxBox.(StatsBase.reconstruct(ptmn90.utZ, Z′); λ = ptmn90.lambda );
        return DataFrame(Float64.([Y;Z])', [ptmn90.paramsY ; ptmn90.paramsZ])
    end;

end;

# ╔═╡ 1d30714c-50fa-11eb-30fc-c9b7e9ad1017
begin	
	pmos90path = "../model/ptmp90-2021-02-10T11:47:20.406/ptmp90.bson"
	pmos90file = BSON.load(pmos90path);

    ptmp90 = MOSFET( pmos90file[:model]
                   , String.(pmos90file[:paramsX])
                   , String.(pmos90file[:paramsY])
                   , String.(pmos90file[:paramsZ])
                   , pmos90file[:utX]
                   , pmos90file[:utY]
                   , pmos90file[:utZ]
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

# ╔═╡ 7d0501a4-54c3-11eb-3e5c-275701e033ea
md"""
## Exploring the Device Model
"""

# ╔═╡ 04fa02b2-50fa-11eb-0a2b-637636057e67
begin
    vgs     = 0.0:0.01:1.2;
    qvgs    = vgs.^2.0;
    vds     = 0.0:0.01:1.2;
    evds    = ℯ.^(vds);
    len     = length(vgs);

end;

# ╔═╡ cca609e0-6b95-11eb-0b6f-fb4542ef729f
#plot( plot( vgs, nmosOPt.id
#          ; xaxis = "Vgs [V]", yaxis = "Id [A]"
#          , title = "Transfer Char @ Vds = $(VD)"
#          , yscale = :lin )
#    , plot( vds, nmosOPo.id
#          ; xaxis = "Vds [V]", yaxis = "Id [A]"
#          , title = "Output Char @ Vgs = $(VG)"
#          , yscale = :lin )
#    ; legend = false
#    , layout = (1,2)
#    , w = 2 )

# ╔═╡ f7d15572-50f9-11eb-2923-fbb666503e9d
begin
	sliderVD = @bind VD Slider( 0.00 : 0.01 : 1.20
							  , default = 0.6, show_value = true );
	sliderVG = @bind VG Slider( 0.00 : 0.01 : 1.20
							  , default = 0.6, show_value = true );
	sliderVB = @bind VB Slider( -1.20 : 0.01 : 0.00
							  , default = 0.0, show_value = true );
	sliderW = @bind W Slider( 1.0e-6 : 0.5e-7 : 10.0e-6
						, default = 5.0e-7, show_value = true );
	sliderL = @bind L Slider( 2e-7 : 1.0e-7 : 2.0e-6
						, default = 3e-7, show_value = true );
	
	md"""
	`Vds` = $(sliderVD) V  `Vgs` = $(sliderVG) V `Vbs` = $(sliderVB) V
	
	`W` = $(sliderW) m `L` = $(sliderL) m
	"""
end

# ╔═╡ 35503cec-6b94-11eb-1ef9-e9d997a3c62d
begin
    w       = fill(W, len);
    l       = fill(L, len);
    vgc     = fill(VG, len);
    qvgc    = vgc.^2.0;
    vdc     = fill(VD, len);
    evdc    = ℯ.^(vdc);
    vbc     = fill(VB, len);
    rvbc    = abs.(vbc).^0.5;
end;

# ╔═╡ 943494f4-6b96-11eb-2972-f91350e19f1f
begin
    xt = [ w l vgs qvgs vdc evdc vbc rvbc ]';
    xo = [ w l vgc qvgc vds evds vbc rvbc ]';
end;

# ╔═╡ e0daee3e-6b96-11eb-22d7-cd70db966684
begin
    nmosOPt = nmos90(xt);
    nmosOPo = nmos90(xo);
end;

# ╔═╡ d7d746b4-6c8d-11eb-2d78-1d27b1d9f7d6
begin
    VGD = hcat(vec([[i,j] for i = 0.0:0.01:1.2, j = 0.0:0.01:1.2])...);
    VGS = VGD[2,:];
    QVGS = VGS.^2.0;
    VDS = VGD[1,:];
    EVDS = ℯ.^VDS;
    LEN = length(VGS);
    VBS = fill(VB, LEN);
    RVBS= abs.(VBS).^0.5;
    WS = fill(W, LEN);
    LS = fill(L, LEN);

    xot = [ WS LS VGS QVGS VDS EVDS VBS RVBS ]';

	yot = nmos90(xot);

	nmos90id = reshape( yot.id, (Int(sqrt(LEN)), Int(sqrt(LEN))));	
	surface( unique(VGS), unique(VDS), nmos90id
           ; c = :blues, legend = false
           , xaxis = "Vgs [V]", yaxis = "Vds [V]", zaxis = "Id [A]" 
           , title = "Characteristic" )
end

# ╔═╡ 46da2f52-5b1c-11eb-3476-73665e0f2366
md"""
### Partial Derivatives
"""

# ╔═╡ 5291467a-5b1d-11eb-1904-0344a8ac78cf
begin
	vLen = 121;
	#pV = range(0.0, length = vLen, stop = 1.2) |> collect;
	#sV = [repeat(pV, inner=[size(pV, 1)]) repeat(pV, outer=[size(pV,1)])]
	#pVgs, pVds = eachcol(sV);
	#pLen = first(size(sV));
	
	pVgs = fill(0.6, vLen);
	pVds = range(0.0, length = vLen, stop = 1.2) |> collect;
	
	pW = fill(2.0e-6, vLen);
	pL = fill(3.0e-7, vLen);
	
	pX = [ pVgs |> collect |> adjoint
		 ; pVds |> collect |> adjoint
		 ; pW |> collect |> adjoint
		 ; pL |> collect |> adjoint ];
end;

# ╔═╡ 8e79cbc0-5bdc-11eb-06a5-29dcd6b17399
function drainCurrent(vgs, vds, w, l)
	pY = predictₙ( [ vgs'  ; (vgs.^2)'
				   ; vds' ; (ℯ.^vds)'
				   ; w' ; l' ] );
	return pY[first(indexin(["id"], paramsYₙ)), :][1];
end;

# ╔═╡ bed655ac-5bdb-11eb-34db-53f5a73d8f1b
pId = [ drainCurrent(x...) for x in eachcol(pX) ];

# ╔═╡ d0eae564-5bc0-11eb-1684-b9973a048479
begin
	plot(pVds, pId)
	#surface!(pVgs, pVds, ∂id_∂LGrid; c = :blues, legend = false)
	#surface!(pVgs, pVds, ∂id_∂WGrid; c = :greens, legend = false)
end

# ╔═╡ 944ca788-5bbf-11eb-100e-bb1ed0c52faf
#begin
	#∂id = hcat([ grad(central_fdm(5,1), id, px)[1] for px in eachcol(pX) ]...);
	#∂id_∂Vgs, ∂id_∂Vgs, ∂id_∂W, ∂id_∂L = eachrow(∂id);
	#∂id_∂LGrid = reshape(∂id_∂L, (Int(sqrt(pLen)), Int(sqrt(pLen))));
	#∂id_∂WGrid = reshape(∂id_∂W, (Int(sqrt(pLen)), Int(sqrt(pLen))));
#end

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
# ╠═ad021952-6b91-11eb-03b5-4daf591847cd
# ╟─a12acdfe-50ff-11eb-018f-df481ecc63df
# ╠═e051a73a-50f9-11eb-3a7a-11906e8f508d
# ╠═1d30714c-50fa-11eb-30fc-c9b7e9ad1017
# ╟─7d0501a4-54c3-11eb-3e5c-275701e033ea
# ╠═04fa02b2-50fa-11eb-0a2b-637636057e67
# ╠═35503cec-6b94-11eb-1ef9-e9d997a3c62d
# ╠═943494f4-6b96-11eb-2972-f91350e19f1f
# ╠═e0daee3e-6b96-11eb-22d7-cd70db966684
# ╠═cca609e0-6b95-11eb-0b6f-fb4542ef729f
# ╟─d7d746b4-6c8d-11eb-2d78-1d27b1d9f7d6
# ╟─f7d15572-50f9-11eb-2923-fbb666503e9d
# ╟─46da2f52-5b1c-11eb-3476-73665e0f2366
# ╠═d0eae564-5bc0-11eb-1684-b9973a048479
# ╠═5291467a-5b1d-11eb-1904-0344a8ac78cf
# ╠═8e79cbc0-5bdc-11eb-06a5-29dcd6b17399
# ╠═bed655ac-5bdb-11eb-34db-53f5a73d8f1b
# ╠═944ca788-5bbf-11eb-100e-bb1ed0c52faf
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
