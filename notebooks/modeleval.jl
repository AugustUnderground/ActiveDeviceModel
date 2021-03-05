### A Pluto.jl notebook ###
# v0.12.21

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
      StatsBase, NNlib, FiniteDifferences, DataFrames, Optim, LineSearches,
      NumericIO, LaTeXStrings, Chain, ForwardDiff, DataInterpolations

# ╔═╡ c7e55070-50f9-11eb-29fa-fd2d4ce6db29
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
	Core.eval(Main, :(using StatsBase))
	Core.eval(Main, :(using ForwardDiff))
end

# ╔═╡ 69d38de8-50fa-11eb-1e3d-4bc49627f622
md"""
# Machine Learning Model Evaluation

## Imports and Setup
"""

# ╔═╡ da09d154-50f9-11eb-3ee8-9112221e4658
#pyplot();
#inspectdr();
plotly();
#pgfplotsx();

# ╔═╡ db81ef56-75a4-11eb-38cf-3577d332d43f
md"""
#### Boilerplate setup for Device Models
"""

# ╔═╡ 3950a4ca-6b90-11eb-3d01-99f0fa677fe4
struct DEVICE
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

# ╔═╡ bf400218-75ae-11eb-25a7-43b64b60e8d1
function loadDevice(fileName)
	deviceFile = BSON.load(fileName);

    dev = DEVICE( deviceFile[:model]
                , String.(deviceFile[:paramsX])
                , String.(deviceFile[:paramsY])
                , deviceFile[:utX]
                , deviceFile[:utY]
                , deviceFile[:maskX]
                , deviceFile[:maskY]
                , deviceFile[:lambda] );

    function operatingPoint(Vgs, Vds, Vbs, W, L)
        input = adjoint(hcat( W, L, Vgs, Vgs.^2.0, Vds
                            , exp.(Vds), Vbs, sqrt.(abs.(Vbs))));
        #input = DataFrame(valuesX, dev.paramsX);
        X = Matrix(input);
        X[dev.maskX,:] = boxCox.(abs.(X[dev.maskX,:]); λ = dev.lambda);
        X′ = StatsBase.transform(dev.utX, X);
        Y′ = dev.model(X′);
        Y = StatsBase.reconstruct(dev.utY, Y′);
        Y[dev.maskY,:] = coxBox.(Y[dev.maskY,:]; λ = dev.lambda);
        return DataFrame(Float64.(Y'), dev.paramsY)
    end;
    
    return dev,operatingPoint
end;

# ╔═╡ 3c723e06-75af-11eb-038b-230378ee15fb
function loadOperator(fileName)
	operatorFile = BSON.load(fileName);

    op = OPERATOR( operatorFile[:parameter]
                 , operatorFile[:model]
                 , String.(operatorFile[:paramsX])
                 , String.(operatorFile[:paramsY])
                 , operatorFile[:utX]
                 , operatorFile[:utY]
                 , operatorFile[:maskX]
                 , operatorFile[:maskY]
                 , operatorFile[:lambda] );

    function operator(l, id, param, vds, vbs)
        input = adjoint(hcat(l, id, param, vds, exp.(vds), vbs, sqrt.(abs.(vbs))));
        X = Matrix(input);
        X[op.maskX,:] = boxCox.( abs.(X[op.maskX,:])
                               ; λ = op.lambda);
        X′ = StatsBase.transform(op.utX, X);
        Y′ = op.model(X′);
        Y = StatsBase.reconstruct(op.utY, Y′);
        Y[op.maskY,:] = coxBox.( Y[op.maskY,:]; λ = op.lambda);
        return DataFrame(Float64.(Y'), op.paramsY)
    end;

    return op,operator
end;

# ╔═╡ a12acdfe-50ff-11eb-018f-df481ecc63df
md"""
## Loading Models

Specifying 90nm Predicitve Technology Model Files
"""

# ╔═╡ 945b2fec-75af-11eb-0b35-e9a5a6ce4e9d
begin
	nmos, φₙ = loadDevice("../model/current/op-nmos90.bson");
	pmos, φₚ = loadDevice("../model/current/op-pmos90.bson");

    opνn, νₙ = loadOperator("../model/current/l-vdsat-nmos90.bson");
    opνp, νₚ = loadOperator("../model/current/l-vdsat-pmos90.bson");
    opγn, γₙ = loadOperator("../model/current/l-gmid-nmos90.bson");
    opγp, γₚ = loadOperator("../model/current/l-gmid-pmos90.bson");
end;

# ╔═╡ 7d0501a4-54c3-11eb-3e5c-275701e033ea
md"""
## Operating Point Model


$$\begin{bmatrix}
V_{\text{GS}} \\
V_{\text{DS}} \\
V_{\text{BS}} \\
W \\
L \\
\end{bmatrix}
\mapsto
\begin{bmatrix}
i_{\text{d}} \\
g_{\text{m}} \\
g_{\text{ds}} \\
f_{\text{ug}} \\
v_{\text{dsat}} \\
c_{\text{xx}} \\
\vdots
\end{bmatrix}$$

"""

# ╔═╡ f7d15572-50f9-11eb-2923-fbb666503e9d
begin
    sliderOPb = @bind opVB Slider( -1.20 : 0.01 : 0.00
                                 , default = 0.0, show_value = false );
    sliderOPw = @bind opW Slider( 0.5e-6 : 0.5e-7 : 10.0e-6
                                , default = 2.0e-6, show_value = false );
    sliderOPl = @bind opL Slider( 1.0e-7 : 1.0e-7 : 2.0e-6
                                , default = 2.0e-7, show_value = false );

md"""
`W` = $(sliderOPw) $(formatted(opW, :ENG, ndigits = 2)) m 
`L` = $(sliderOPl) $(formatted(opL, :ENG, ndigits = 2)) m

`Vbs` = $(sliderOPb) $(formatted(opVB, :ENG, ndigits = 2)) V
"""
end

# ╔═╡ e702caf0-720f-11eb-2780-1770c60d8247
begin
    #L = rand(1e-7:1e-5);
    #W = rand(1e-6:1e-4);
    #VGD = hcat(vec([[i,j] for i = 0.0:0.01:1.2, j = 0.0:0.01:1.2])...);
    #VGS = VGD[2,:];
    #VDS = VGD[1,:];
	VGS = 0:0.01:1.2;
	VDS = fill(0.6, 121);
    LEN = length(VGS);
    VBS = fill(opVB, LEN);
    WS = fill(opW, LEN);
    LS = fill(opL, LEN);

    nmosOPt = φₙ(VGS, VDS, VBS, WS, LS).id;

    #nmos90id = reshape( nmosOPt.id, (Int(sqrt(LEN)), Int(sqrt(LEN))));	

    # surface( unique(VGS), unique(VDS), nmos90id
    #        ; c = :thermal, legend = false
    #        , xaxis = "Vgs [V]", yaxis = "Vds [V]", zaxis = "Id [A]" 
    #        , title = "Characteristic" )
	plot(VGS, nmosOPt, yscale = :log10)
end

# ╔═╡ b7af167c-7b66-11eb-3a89-fdf505c25551
VGS

# ╔═╡ e1d7d448-72c9-11eb-3ea6-99c8336551de
md"""
## Design Models


$$\nu : \begin{bmatrix}
L \\
i_{\text{d}} \\
v_{\text{dsat}} \\
V_{\text{DS}} \\
V_{\text{BS}} \\
\end{bmatrix}
\mapsto
\begin{bmatrix}
W \\
V_{\text{GS}} \\
g_{\text{ds}} \\
g_{\text{m}} \\
f_{\text{ug}} \\
\vdots
\end{bmatrix}$$
$$\gamma : \begin{bmatrix}
L \\
i_{\text{d}} \\
g_{\text{m}} / i_{\text{d}} \\
V_{\text{DS}} \\
V_{\text{BS}} \\
\end{bmatrix}
\mapsto
\begin{bmatrix}
W \\
V_{\text{GS}} \\
g_{\text{ds}} \\
v_{\text{dsat}} \\
f_{\text{ug}} \\
\vdots
\end{bmatrix}$$
"""

# ╔═╡ aa26b788-72cb-11eb-3508-05c306d752d2
begin
    sliderDVl = @bind dL Slider( 1e-7 : 1e-7 : 1e-6
                              , default = 3e-7, show_value = true );
    sliderDVi = @bind dI Slider( 1e-6 : 1e-6 : 100e-6
                               , default = 25e-6, show_value = true );
    sliderDVd = @bind dD Slider( 0.00 : 0.01 : 1.20
                               , default = 0.6, show_value = true );
    sliderDVb = @bind dB Slider( -1.00 : 0.01 : 0.00
                               , default = 0.0, show_value = true );

md"""
`L` = $(sliderDVl) m
`Id` = $(sliderDVi) A

`Vds` = $(sliderDVd) V
`Vbs` = $(sliderDVb) V

"""
end

# ╔═╡ 9796974e-75a7-11eb-2b81-fde8bed16a29
begin
    idxJd = first(indexin(["Jd"], opνn.paramsY));
    idxA0 = first(indexin(["A0"], opνn.paramsY));

    dLen = 6;
    sweepᵥ = collect(range(0.05, stop = 0.5, length = dLen));
    sweepᵧ = collect(range(5, stop = 25, length = dLen));
    
    dYν = vcat(map((vdsat) -> νₙ(dL, dI, vdsat, dD, dB), sweepᵥ)...);
    dYγ = vcat(map((gmid) -> γₙ(dL, dI, gmid, dD, dB), sweepᵧ)...);

    Jdν = CubicSpline(Vector(dYν.Jd), sweepᵥ);
    A0ν = CubicSpline(Vector(dYν.A0), sweepᵥ);
    Jdγ = CubicSpline(Vector(dYγ.Jd), sweepᵧ);
    A0γ = CubicSpline(Vector(dYγ.A0), sweepᵧ);
end;

# ╔═╡ c96a159e-75b0-11eb-14b8-05e3b18a842c
plot( plot(Jdν; yaxis = "Jd", legend = false, yscale = :log10)
    , plot(Jdγ; yaxis = "Jd", legend = false, yscale = :log10)
    , plot(A0ν; xaxis = "vdsat", yaxis = "A0", legend = false, yscale = :log10)
    , plot(A0γ; xaxis = "gm/id", yaxis = "A0", legend = false, yscale = :log10)
    ; layout = (2,2) )

# ╔═╡ ecb93d9e-75b1-11eb-1062-1f0dad69da98
begin
    ∇A0(vdsat) = first(Zygote.gradient(A0ν, vdsat));
    ∂A0 = ∇A0.(sweepᵥ)
    ∇Jd(vdsat) = first(Zygote.gradient(Jdγ, vdsat));
    ∂Jd = ∇Jd.(sweepᵥ)
end;

# ╔═╡ 87c15636-76bc-11eb-0985-99a62dc1c233
md"""
---
## Example: Symmetrical Amplifier

### Specification
"""

# ╔═╡ fb8d653c-75d1-11eb-27a1-fbac99f77c07
begin
    VDD  = 1.2;
    VSS  = 0.0;
    Vicm = 0.5;
    Vocm = 0.6;
    Lmin = 3.0e-7;
    Lmax = 3.0e-6;
end;

# ╔═╡ d3a287be-75d6-11eb-35a4-732cd461df58
begin
    sliderDEIR = @bind Iref Slider( 10e-6 : 5e-6 : 100e-6
                                  , default = 50e-6, show_value = true );
    sliderDECL = @bind CL Slider( 1.0e-12 : 1.0e-12 : 25e-12
                                  , default = 10e-12, show_value = true );
md"""
**Spec:**

| Iref | CL  |
|:----:|:---:|
| $(sliderDEIR) A | $(sliderDECL) F |
"""
end

# ╔═╡ 98136d66-75d7-11eb-1bbe-159e5f431357
begin
    sliderDEL12 = @bind L12 Slider( 1.0e-7 : 1.0e-7 : 1.0e-6
                                  , default = 3e-7, show_value = true );
    sliderDEL34 = @bind L34 Slider( 1.0e-7 : 1.0e-7 : 1.0e-6
                                  , default = 3e-7, show_value = true );
    sliderDEL78 = @bind L78 Slider( 1.0e-7 : 1.0e-7 : 1.0e-6
                                  , default = 3e-7, show_value = true );
    sliderDEL90 = @bind L90 Slider( 1.0e-7 : 1.0e-7 : 1.0e-6
                                  , default = 3e-7, show_value = true );
    

    sliderDEM   = @bind M Slider(1.0:1.0:10.0
                                , default = 5, show_value = true );
    sliderDEK   = @bind K Slider(0.1:0.1:1.0
                                , default = 0.5, show_value = true );
    
    #sliderDEv12 = @bind vdsat12 Slider( 0.05 : 0.01 : 0.5
    #                                  , default = 0.2, show_value = true );
    
    sliderDEg12 = @bind gmid12 Slider( 5 : 1 : 20
                                      , default = 12, show_value = true );
    sliderDEv34 = @bind vdsat34 Slider( 0.05 : 0.01 : 0.5
                                      , default = 0.2, show_value = true );
    sliderDEv78 = @bind vdsat78 Slider( 0.05 : 0.01 : 0.5
                                      , default = 0.2, show_value = true );
    sliderDEv90 = @bind vdsat90 Slider( 0.05 : 0.01 : 0.5
                                      , default = 0.2, show_value = true );
md"""
**Design:**

|  M  |  K  |
|:---:|:---:|
| $(sliderDEM) |$(sliderDEK) |

| Design Parameter |   Value   | Sizing  | Length |
|:-----------------|:---------:|:--------|:------:|
| gmid12  | $(sliderDEg12) S/A | L12 | $(sliderDEL12) m |
| vdsat34 | $(sliderDEv34) V   | L34 | $(sliderDEL34) m |
| vdsat78 | $(sliderDEv78) V   | L78 | $(sliderDEL78) m |
| vdsat90 | $(sliderDEv90) V   | L90 | $(sliderDEL90) m |
"""
end

# ╔═╡ 37ee6758-7747-11eb-3055-31b667b966ef
md"""
#### 1. NMOS Current Mirror CM4

Evaluating 

$$\nu_{n} ( L_{7,8}, I_{6,8}, v_{\text{dsat},7,8}, V_{out,cm}, 0.0 )$$

to obtain sizing for $NM_{\text{cm,41}}$ and $NM_{\text{cm,42}}$.

"""

# ╔═╡ 3d6ff8da-7748-11eb-1d9c-ab6c3e3618fa
begin
    I68 = ((K * M) / 2 ) * Iref;
    MN8 = νₙ(L78, I68, vdsat78, Vocm, 0.0);
    Vgs8, Jd8, gds8 = Matrix(MN8[:,[:Vgs, :Jd, :gds]]);
    W78 = I68 / Jd8;
	
    Vd = Vgs8;
end;

# ╔═╡ af57da44-7748-11eb-0b20-fbd45045a28a
md"""
#### 2. PMOS Current Mirrors CM2 and CM3

Evaluating

$$\nu_{p} ( L_{3,4}, I_{6,8}, v_{\text{dsat},3,4}, (V_{dd} - V_{out,cm}), 0.0 )$$

to obtain sizing for $PM_{\text{cm,21}}$ and $PM_{\text{cm,22}}$, 
as well as $PM_{\text{cm,31}}$ and $PM_{\text{cm,32}}$.
"""

# ╔═╡ de5a675a-7748-11eb-2abd-9d75649902f8
begin
    MP6 = νₚ(L34, I68, vdsat34, (VDD - Vocm), 0.0);
    Vgs6, Jd6, gds6 = Matrix(MP6[:,[:Vgs, :Jd, :gds]]);
    W34 = (I68 / Jd6) / M
    W65 = (W34 * M) + ((W34) / 10.0);
    Vb = Vc = (VDD - Vgs6);
	
    #I34 = I68 / M;
    #MP4 = νₚ(L34, I34, vdsat34, (VDD - Vb), 0.0);
    #W4, gds4, Vgs4, Jd4 = Matrix(MP4[:,[:W, :gds, :Vgs, :Jd]]);
end;

# ╔═╡ f9a7f5be-7748-11eb-39bc-0f74d0dd0453
md"""
#### 3. Differential Pair


Evaluating

$$\gamma_{n} ( L_{1,2}, I_{1,2}, (g_{\text{m}} / i_{\text{d}})_{3,4}
             , (V_{D} - V_{A}), - V_{A} )$$

to obtain sizing for $NM_{\text{diff,1}}$ and $NM_{\text{diff,2}}$, 
"""

# ╔═╡ 166a48ca-7749-11eb-02c4-0704b6909715
begin
    I12 = (K / 2) * Iref;
    Va = Vc / 3;
    
	MN1 = γₙ(L12, I12, gmid12, (Vc - Va), -Va);

    Vgs1, Jd1, gm1 = Matrix(MN1[:,[:Vgs, :Jd, :gm]]);
    W12 = I12 / Jd1;
end;

# ╔═╡ 2a7c0dba-7749-11eb-20cd-210ab922d0af
md"""
#### 4. Biasing Current Mirror CM1


Evaluating 

$$\nu_{n} ( L_{9,0}, I_{9,0}, v_{\text{dsat},9,0}, V_{A}, 0.0 )$$

to obtain sizing for $NM_{\text{cm,11}}$ and $NM_{\text{cm,12}}$.

"""

# ╔═╡ 42657a58-7749-11eb-063b-19ed798a2c95
begin
    I90 = K * Iref;
    MN9 = νₙ(L90, I90, vdsat90, Va, 0.0);
    W9, Vgs9, Jd9 = Matrix(MN9[:,[:W, :Vgs, :Jd]]);
    W0 = (I90 / Jd9) / K;
    W9 = (W0 * K) + ((W0 * K) / 10.0);
end;

# ╔═╡ 7d0392d8-7746-11eb-34a8-3508ea7402cd
md"""
#### Circuit Performance 

**DC Gain:**
$$A_{0} = M \cdot g_{\text{m},1} \cdot r_{\text{out}}$$

**Cutoff Frequency:**
$$f_{0} = ( 2 \pi \cdot C_{\text{L}} \cdot r_{\text{out}} )^{-1}$$

**Slew Rate:**
$$\text{SR} = \frac{M \cdot K \cdot I_{\text{ref}}}{2 \cdot C_{\text{L}}}$$
"""

# ╔═╡ 5215f098-7746-11eb-0c16-a1a3b9c5ae81
begin
    rout = 1 / (gds6 + gds8);

    A0 = M * gm1 * rout ;
    A0dB = 20 * log10( abs( A0) );

    f0 = 1 / (2π * CL * rout);
    ω0 = 1 / (CL * rout)
    
    GBW = A0 * f0;
    
    SR = I68 / CL;
end;

# ╔═╡ 3dbf3974-54fa-11eb-1901-c310b0dd8685
begin
#$(PlutoUI.LocalResource("./symamp.png"))
md"""
### Summary

![](https://gitlab-forschung.reutlingen-university.de/electronics-and-drives/tikzlib/-/raw/master/img/schematic_sym-opamp.png)

**Specification:**

| Parameter | Value                         |
|:----------|------------------------------:|
| Vdd       | $(formatted(VDD, :ENG; ndigits = 5)) V     |
| Iref      | $(formatted(Iref, :ENG; ndigits = 5)) A    |
| Vincm     | $(formatted(Vicm, :ENG; ndigits = 5)) V    |
| Vocm      | $(formatted(Vocm, :ENG; ndigits = 5)) V    |
| CL        | $(formatted(CL, :ENG; ndigits = 5)) F      |
| Lmin      | $(formatted(Lmin, :ENG; ndigits = 5)) m    |
| Lmax      | $(formatted(Lmax, :ENG; ndigits = 5)) m    |

**Sizing Parameters:**

| Parameter | Value                             |
|:----------|----------------------------------:|
| L1, L2    | $(formatted(L12, :ENG; ndigits = 5)) m    |
| L3, L4, L5, L6   | $(formatted(L34, :ENG; ndigits = 5)) m    |
| L7, L8    | $(formatted(L78, :ENG; ndigits = 5)) m    |
| L9, L0    | $(formatted(L90, :ENG; ndigits = 5)) m    |
| W1, W2    | $(formatted(W12, :ENG; ndigits = 5)) m    |
| W3, W4    | $(formatted(W34, :ENG; ndigits = 5)) m    |
| W5, W6    | $(formatted(W65, :ENG; ndigits = 5)) m    |
| W7, W8    | $(formatted(W78, :ENG; ndigits = 5)) m    |
| W9        | $(formatted(W9, :ENG; ndigits = 5)) m    |
| W0        | $(formatted(W0, :ENG; ndigits = 5)) m    |

**Performance:**

| Parameter  | Value |
|:-----------|------:|
| DC Gain    | $(formatted(A0dB, :ENG; ndigits = 5)) dB |
| 3dB Cutoff | $(formatted(f0, :ENG; ndigits = 5)) Hz |
| GBW  | $(formatted(GBW, :ENG; ndigits = 5)) |
| SR  | $(formatted(SR, :ENG; ndigits = 5)) |
"""
end

# ╔═╡ e703b8fc-776e-11eb-19f8-cd6b4888e233
begin
    amp(ω) = 20 * log10(abs(A0) / sqrt(1 + (ω / ω0)^2));
    ph(ω) = -atan(ω / ω0) * 180 / π;

    ω = exp10.(range(1, stop = 9, length = 50));

    ampli = amp.(ω);
    phase = ph.(ω);

    plot( plot(ω, ampli; title = "Frequency Response", yaxis = "A₀ [dB]", xscale = :log10)
        , plot(ω, phase; xaxis = "ω [1/s]", yaxis = "arg(A₀)", xscale = :log10)
        ; layout = (2,1), legend = false )
end

# ╔═╡ 9534399a-774e-11eb-35f0-15e1ac87ce3e
md"""
---
### Optimization Attempt

While the intended use of this methodology is either in form of a script
or interactively, it can be *abused* with an optimizer.

#### Symmetrical Amplifier as a single Function
"""

# ╔═╡ b4e1b330-774e-11eb-2e19-61f86b046c52
function symamp(L12, L34, L78, L90, gmid12, vdsat34, vdsat78, vdsat90)
    I68 = ((K * M) / 2 ) * Iref;

    MN8 = νₙ(L78, I68, vdsat78, Vocm, 0.0);
    gds8, Vgs8, Jd8 = Matrix(MN8[:,[:gds, :Vgs, :Jd]]);
    W78 = I68 / Jd8;

    Vd = Vgs8;
	
    MP6 = νₚ(L34, I68, vdsat34, Vocm, 0.0);
    gds6, Vgs6, Jd6 = Matrix(MP6[:,[:gds, :Vgs, :Jd]]);
    W34 = (I68 / Jd6) / M
    W56 = (W34 * M) + ((W34) / 10.0);

    Vb = Vc = (VDD - Vgs6);
    I12 = (K / 2) * Iref;
    Va = Vc / 3;
	
    MN1 = γₙ(L12, I12, gmid12, (Vb - Va), -Va);
    #MN1 = νₙ(L12, I12, vdsat12, VDD/3, 0.0);
    #MN1 = γₙ(L12, I12, gmid12, Va, 0.0);
    gm1, Jd1 = Matrix(MN1[:,[:gm, :Jd]]);
    W12 = I12 / Jd1;
	
    I90 = K * Iref;

    MN9 = νₙ(L90, I90, vdsat90, Va, 0.0);
    Vgs9, Jd9 = Matrix(MN9[:,[:Vgs, :Jd]]);
    W0 = (I90 / Jd9) / K;
    W9 = (W0 * K) + ((W0 * K) / 10.0);
	
    rout = 1 / (gds6 + gds8);
    A₀ = M * gm1 * rout ;
    A₀dB = 20 * log10(abs(A₀));
    #ω₀ = 1 / (CL * rout);
    f₀ = 1 / (2π * CL * rout);
    return [A₀dB, f₀, W12, W34, W56, W78, W9, W0]
end;

# ╔═╡ 1980a07a-7752-11eb-22d2-31964e897bfc
md"""
#### Objective Functions

**Gain:**
"""

# ╔═╡ dcc8b0ce-7767-11eb-057e-75ecc6425e21
md"""
**Bandwidth:**
"""

# ╔═╡ d9bd718a-7668-11eb-3d78-1b1f50ba7227
begin
    normL(L) = (L - Lmin) / (Lmax - Lmin);
    realL(L′) = L′ * (Lmax - Lmin) + Lmin;
    
	Linit = normL(3Lmin);
end;

# ╔═╡ a8351c5a-7765-11eb-06c3-4d1f79b686df
md"""
Boundaries and initial condition:
"""

# ╔═╡ 9c6b3eea-7765-11eb-1bd4-59a0ab8af5d3
begin
    lower = zeros(3);
    upper = ones(3);
	init = rand(3);
	
    
    #optimAlgorithm = ConjugateGradient();
    optimAlgorithm = GradientDescent();
	#optimAlgorithm = Newton();
    #optimAlgorithm = GradientDescent(linesearch=LineSearches.BackTracking(order=2));
    #optimAlgorithm = LBFGS();
	
    optimOptions = Optim.Options( g_tol = 1e-8
                                , x_tol = 1e-8
                                , f_tol = 1e-8
                                , time_limit = 25.0 );
	
	vdsat12Fix = vdsat34Fix = vdsat78Fix = 0.2;
	vdsat90Fix = 0.11;
	gmid12Fix = gmid34Fix = gmid78Fix = gmid90Fix = 15;
    
    L90Fix = 700e-9;
end;

# ╔═╡ 08f2dda2-7752-11eb-2000-95f26e32ed60
begin
    A0dBtarget = 50.0;

md"""
**Objectvie Gain:** $A_{0,target} =$ $(formatted(A0dBtarget, :ENG; ndigits = 3)) dB
"""
end

# ╔═╡ b310ed0a-7767-11eb-3380-e10c44018165
function symampGainObjective(X)
	L12D, L34D, L78D = realL.(X);
    
    A0dB, _ = symamp( L12D, L34D, L78D, L90Fix
					, gmid12Fix, vdsat34Fix, vdsat78Fix, vdsat90Fix);
    loss = abs(A0dB - A0dBtarget)
	
    return loss
end;

# ╔═╡ 19154a02-7750-11eb-0165-1f6bef4d70ea
gainResults = optimize( symampGainObjective, lower, upper, init
                      , Fminbox(optimAlgorithm), optimOptions)

# ╔═╡ 671d8e50-7765-11eb-1b2b-f1d37f61b068
begin
    f0target = 65e3;

md"""
**Objectvie Bandwidth:** $f_{0,target} =$ $(formatted(f0target, :ENG; ndigits = 3)) Hz
"""
end

# ╔═╡ 899dec4c-774f-11eb-1c6e-01c49eacb865
function symampBWObjective(X)
	L12D, L34D, L78D = realL.(X);
	
    _, f0, _ = symamp( L12D, L34D, L78D, L90Fix
					 , gmid12Fix, vdsat34Fix, vdsat78Fix, vdsat90Fix);
	
    loss = abs(f0 - f0target)
	
    return loss
end;

# ╔═╡ 3af0e86c-7766-11eb-2499-77082a798dbb
bwResults = optimize( symampBWObjective, lower, upper, init
                    , Fminbox(optimAlgorithm), optimOptions)

# ╔═╡ 17ccd0fe-783c-11eb-14e2-bf3afe3f5edb
begin
    gL12, gL34, gL78 = realL.(gainResults.minimizer);

    gSizing = symamp( gL12, gL34, gL78, L90
                   , gmid12Fix, vdsat34Fix, vdsat78Fix, vdsat90Fix );
    gA0dB, gf0, gW12, gW34, gW56, gW78, gW9, gW0 = gSizing;

end;

# ╔═╡ 480bfa84-775f-11eb-3d26-ffab3de5f965
begin
    fL12, fL34, fL78 = realL.(bwResults.minimizer);

    fSizing = symamp( fL12, fL34, fL78, L90
                   , gmid12Fix, vdsat34Fix, vdsat78Fix, vdsat90Fix );
    fA0dB, ff0, fW12, fW34, fW56, fW78, fW9, fW0 = fSizing;
end;

# ╔═╡ 4bcb1c1a-7762-11eb-1518-cf9f3d0e13dd
begin
    SgA0dB = 4.597800e+01;
    Sgf0 = 5.570337e+04;
    SfA0dB = 39.04051;
    Sff0 = 1.479150e5;


md"""
#### Results

**Sizing for Optimization:**

| Sizing Parameter | Gain | Bandwidth |
|:-----------------|-----:|----------:|
| L1, L2 | $(formatted(gL12, :ENG; ndigits = 4)) m | $(formatted(fL12, :ENG; ndigits = 4)) m |
| L3, L4, L5, L6   | $(formatted(gL34, :ENG; ndigits = 4)) m | $(formatted(fL34, :ENG; ndigits = 4)) m |
| L7, L8 | $(formatted(gL78, :ENG; ndigits = 4)) m | $(formatted(fL78, :ENG; ndigits = 4)) m |
| L9, L0 | $(formatted(L90, :ENG; ndigits = 4)) m | $(formatted(L90, :ENG; ndigits = 4)) m |
| W1, W2 | $(formatted(gW12, :ENG; ndigits = 4)) m | $(formatted(fW12, :ENG; ndigits = 4)) m |
| W3, W4 | $(formatted(gW34, :ENG; ndigits = 4)) m | $(formatted(fW34, :ENG; ndigits = 4)) m |
| W5, W6 | $(formatted(gW56, :ENG; ndigits = 4)) m | $(formatted(fW56, :ENG; ndigits = 4)) m |
| W7, W8 | $(formatted(gW78, :ENG; ndigits = 4)) m | $(formatted(fW78, :ENG; ndigits = 4)) m |
| W9     | $(formatted(gW9, :ENG; ndigits = 4)) m  | $(formatted(fW9, :ENG; ndigits = 4)) m  |
| W0     | $(formatted(gW0, :ENG; ndigits = 4)) m  | $(formatted(fW0, :ENG; ndigits = 4)) m  |

Using these sizes in an actual Simulation yields:

for **Gain** Optimization:

| Performance Parameter | Prediction | Simulation | $\Delta$ |
|:----------------------|-----------:|-----------:|---------:|
| A0dB | $(formatted(gA0dB, :ENG; ndigits = 4)) dB | $(formatted(SgA0dB, :ENG; ndigits = 4)) dB | $(formatted(gA0dB - SgA0dB, :ENG; ndigits = 4)) |
| f0   | $(formatted(gf0, :ENG; ndigits = 4)) Hz | $(formatted(Sgf0, :ENG; ndigits = 4)) Hz | $(formatted(gf0 - Sgf0, :ENG; ndigits = 4)) |

for **Bandwidth** Optimization:

| Performance Parameter | Prediction | Simulation | $\Delta$ |
|:----------------------|-----------:|-----------:|---------:|
| A0dB | $(formatted(fA0dB, :ENG; ndigits = 4)) dB | $(formatted(SfA0dB, :ENG; ndigits = 4)) dB | $(formatted(fA0dB - SfA0dB, :ENG; ndigits = 4)) |
| f0   | $(formatted(ff0, :ENG; ndigits = 4)) Hz   | $(formatted(Sff0, :ENG; ndigits = 4)) Hz   | $(formatted(ff0 - Sff0, :ENG; ndigits = 4)) |

"""
end

# ╔═╡ 6c83900a-76b2-11eb-30f1-07ef0afeaf98
md"""
---
---
---
# Depracated
"""

# ╔═╡ aaf3d4a8-76b2-11eb-001f-c563037c0b4c
function differentialPair(mosType, Iref, Vgs, Vds, Vbs, gmid)
    φ,γ = mosType == :n ? (φₙ, γₙ) : (φₚ, γₚ);
    Id = (Iref / 2);
    gm0 = gmid * Id;

    metaDP(L′) = begin
        L = realL(L′);
        Jd = first(γ(L, Id, gmid, Vds, Vbs).Jd);
        W = (1 / Jd) * Id;
        gds = first(φ(Vgs, Vds, Vbs, W, L).gds);
        return sqrt(abs((1 / gds) - (1 / 5e-6)))
    end;

    optDP = optimize(metaDP, 0, 1, Brent(); abs_tol = 1e-8, rel_tol = 1e-8);
    
    Ldp = realL(optDP.minimizer);
    Jd, A0, fug, gds, gm, vdsat = Matrix(γ( Ldp, Id, gmid, Vds, Vbs)[:,
								            [:Jd, :A0, :fug, :gds, :gm, :vdsat]]);
    Wdp = (Id / Jd);

    DP = Dict([ "W" => Wdp
              , "L" => Ldp
              , "gds" => gds
              , "gm" => gm
              , "vdsat" => vdsat
              , "Iout" => Id
              , "selfGain" => A0
              , "fug" => fug
              ]);
    
    return DP
end;

# ╔═╡ 02e9348e-76a7-11eb-0749-b18ec7721cd3
function currentMirror(mosType, Iin, Vout, Mratio, vdsat)
    φ,ν = mosType == :n ? (φₙ, νₙ) : (φₚ, νₚ);
    
    Iout = Mratio * Iin;

    metaCM(L′) = begin
        L = realL(L′);
        vgsCM,jdCM = Matrix(ν(L, Iout, vdsat, Vout, 0.0)[:, [:Vgs, :Jd]]);
        wCM = (1 / (jdCM / Iout));
        idCM = first(φ(vgsCM, vgsCM, 0.0, (wCM / Mratio), L).id)
        return sqrt(abs(Iin - idCM))
    end;

    optCM = optimize(metaCM, 0, 1, Brent(); abs_tol = 1e-8, rel_tol = 1e-8);
    
    Lcm = realL(optCM.minimizer);

    JdOut, Vin, gdsOut = Matrix(ν(Lcm, Iout, vdsat, Vout, 0.0)[:, [:Jd, :Vgs, :gds]]);

    Wout = (Iout / JdOut);
    Win = (Wout / Mratio);

    Iin,vdsat = Matrix(φ(Vin, Vin, 0.0, Win, Lcm)[:, [:id, :vdsat]]);
    
    CM = Dict([ "Win" => Win
              , "Wout" => Wout
              , "L" => Lcm
              , "gds" => gdsOut
              , "vdsat" => vdsat
              , "Iin" => Iin
              , "Iout" => Iout 
              , "Vout" => Vout
              , "Vin" => Vin ]);
    
    return CM
end;

# ╔═╡ fe08c92e-76b6-11eb-1d71-e983816941c9
#begin
#    Iout = (K * M / 2) * Iref;
#    Idp = (K / 2 ) * Iref;
#    Va = (VDD / 3);
#
#    Ibias = K * Iref;
#
#    CM78 = currentMirror(:n, Iout, Vocm, 1.0, vdsat78);
#
#    CM46 = CM35 = currentMirror(:p, Idp, Vocm, M, vdsat34);
#
#    CM90 = currentMirror(:n, Iref, Va, K, vdsat90);
#
#    #DP12 = differentialPair(:n, Iref, (Vicm - Va), Va, -Va, vdsat12);
#    DP12 = differentialPair(:n, Ibias, 0.4, 0.5, 0.2, gmid12);
#end;

# ╔═╡ 850130fc-75ee-11eb-1363-67184e15f8ce
#begin
#    Vd = Vocm;
#    Id78 = (M * K * Iref / 2);
#
#    meta78(L′) = begin
#        L = realL(L′);
#        vgs7 = first(νₙ(L, Id78, vdsat78, Vd, 0.0).Vgs);
#        return sqrt(abs(vgs7 - Vd))
#    end;
#
#    opt78 = optimize(meta78, 0, 1, Brent(); abs_tol = 1e-8, rel_tol = 1e-8);
#
#    L7 = L8 = realL(opt78.minimizer);
#
#    NCM78 = νₙ(L7, Id78, vdsat78, Vd, 0.0);
#    Jd78,gds8 = Matrix(NCM78[:,[:Jd, :gds]]);
#    W8 = W7 = 1 / (Jd78 / Id78);
#end;

# ╔═╡ ee915bfa-7670-11eb-27d8-7bc3ea3662ec
#begin
#    Id56 = (M * K * Iref / 2);
#    Id34 = (K * Iref / 2);
#
#    meta34(L′) = begin
#        L = realL(L′);
#        vgs56 = first(νₚ(L, Id56, vdsat34, 0.0, (VDD - Vd)).Vgs);
#        vgs34 = first(νₚ(L, Id34, vdsat34, 0.0, vgs56).Vgs);
#        return sqrt(abs(vgs56 - vgs34))
#    end;
#
#    opt34 = optimize(meta34, 0, 1, Brent(); abs_tol = 1e-8, rel_tol = 1e-8);
#
#    L34 = L56 = realL(opt34.minimizer);
#
#    PCM56 = νₙ(L56, Id56, vdsat34, 0.0, (VDD - Vd));
#    Jd56,gds6,vgs56 = Matrix(PCM56[:,[:Jd, :gds, :Vgs]]);
#    Vb = Vc = (VDD - vgs56);
#    W5 = W6 = 1 / (Jd56 / Id56);
#
#    PCM34 = νₙ(L34, Id34, vdsat34, 0.0, (VDD - Vb) );
#    W3 = W4 = 1 / (PCM34.Jd[1] / Id34);
#end;

# ╔═╡ 415fc368-767a-11eb-0dcb-37b5fce5eb15
#begin
#    Va = (VDD / 3);
#    Id9 = K * Iref;
#    Id0 = Iref;
#
#    meta90(L′) = begin
#        L = realL(L′);
#        vgs9 = first(νₙ(L, Id9, vdsat90, 0.0, Va).Vgs);
#        vgs0 = first(νₙ(L, Id0, vdsat90, 0.0, vgs9).Vgs);
#        return sqrt(abs(vgs0 - vgs9))
#    end;
#
#    opt90 = optimize(meta90, 0.0, 1.0, Brent(); abs_tol = 1e-8, rel_tol = 1e-8);
#
#    L9 = L0 = realL(opt90.minimizer);
#    NCM9 = νₙ(L9, Id9, vdsat90, 0.0, Va);
#    Vi = NCM9.Vgs[1];
#    NCM0 = νₙ(L0, Id0, vdsat90, 0.0, Vi);
#    W9 = 1 / (NCM9.Jd[1] / Id9);
#    W0 = 1 / (NCM0.Jd[1] / Id0);
#end;

# ╔═╡ 03cbeb7a-7676-11eb-2c1e-2196b1ee104b
#begin
#    Id12 = (K * Iref / 2);
#
#    meta12(L′) = begin
#        L = realL(L′);
#        vgs12 = first(νₙ(L, Id12, vdsat12, -Va, (Vc - Va)).Vgs);
#        return sqrt(abs(vgs12 - (Vicm - Va)))
#    end;
#
#    opt12 = optimize(meta12, 0, 1, Brent());
#    L1 = L2 = realL(opt12.minimizer);
#
#    NCM1 = νₙ(L1, Id12, vdsat12, -Va, (Vc - Va));
#    Jd12,gm1 = Matrix(NCM1[:,[:Jd, :gm]]);
#    W1 = W2 = 1 / (Jd12 / Id12);
#end;

# ╔═╡ Cell order:
# ╟─69d38de8-50fa-11eb-1e3d-4bc49627f622
# ╠═aa866870-50f9-11eb-268e-8b614dd7f83c
# ╠═c7e55070-50f9-11eb-29fa-fd2d4ce6db29
# ╠═da09d154-50f9-11eb-3ee8-9112221e4658
# ╟─db81ef56-75a4-11eb-38cf-3577d332d43f
# ╠═3950a4ca-6b90-11eb-3d01-99f0fa677fe4
# ╠═39cfd4de-72ca-11eb-1b25-f37f84953389
# ╠═bf400218-75ae-11eb-25a7-43b64b60e8d1
# ╠═3c723e06-75af-11eb-038b-230378ee15fb
# ╠═ad021952-6b91-11eb-03b5-4daf591847cd
# ╟─a12acdfe-50ff-11eb-018f-df481ecc63df
# ╠═945b2fec-75af-11eb-0b35-e9a5a6ce4e9d
# ╟─7d0501a4-54c3-11eb-3e5c-275701e033ea
# ╟─e702caf0-720f-11eb-2780-1770c60d8247
# ╟─f7d15572-50f9-11eb-2923-fbb666503e9d
# ╠═b7af167c-7b66-11eb-3a89-fdf505c25551
# ╟─e1d7d448-72c9-11eb-3ea6-99c8336551de
# ╟─c96a159e-75b0-11eb-14b8-05e3b18a842c
# ╟─aa26b788-72cb-11eb-3508-05c306d752d2
# ╠═9796974e-75a7-11eb-2b81-fde8bed16a29
# ╠═ecb93d9e-75b1-11eb-1062-1f0dad69da98
# ╟─87c15636-76bc-11eb-0985-99a62dc1c233
# ╠═fb8d653c-75d1-11eb-27a1-fbac99f77c07
# ╟─3dbf3974-54fa-11eb-1901-c310b0dd8685
# ╟─d3a287be-75d6-11eb-35a4-732cd461df58
# ╟─98136d66-75d7-11eb-1bbe-159e5f431357
# ╟─e703b8fc-776e-11eb-19f8-cd6b4888e233
# ╟─37ee6758-7747-11eb-3055-31b667b966ef
# ╠═3d6ff8da-7748-11eb-1d9c-ab6c3e3618fa
# ╟─af57da44-7748-11eb-0b20-fbd45045a28a
# ╠═de5a675a-7748-11eb-2abd-9d75649902f8
# ╟─f9a7f5be-7748-11eb-39bc-0f74d0dd0453
# ╠═166a48ca-7749-11eb-02c4-0704b6909715
# ╟─2a7c0dba-7749-11eb-20cd-210ab922d0af
# ╠═42657a58-7749-11eb-063b-19ed798a2c95
# ╟─7d0392d8-7746-11eb-34a8-3508ea7402cd
# ╠═5215f098-7746-11eb-0c16-a1a3b9c5ae81
# ╟─9534399a-774e-11eb-35f0-15e1ac87ce3e
# ╠═b4e1b330-774e-11eb-2e19-61f86b046c52
# ╟─1980a07a-7752-11eb-22d2-31964e897bfc
# ╠═b310ed0a-7767-11eb-3380-e10c44018165
# ╟─dcc8b0ce-7767-11eb-057e-75ecc6425e21
# ╠═899dec4c-774f-11eb-1c6e-01c49eacb865
# ╠═d9bd718a-7668-11eb-3d78-1b1f50ba7227
# ╟─a8351c5a-7765-11eb-06c3-4d1f79b686df
# ╠═9c6b3eea-7765-11eb-1bd4-59a0ab8af5d3
# ╟─08f2dda2-7752-11eb-2000-95f26e32ed60
# ╠═19154a02-7750-11eb-0165-1f6bef4d70ea
# ╠═671d8e50-7765-11eb-1b2b-f1d37f61b068
# ╠═3af0e86c-7766-11eb-2499-77082a798dbb
# ╠═17ccd0fe-783c-11eb-14e2-bf3afe3f5edb
# ╠═480bfa84-775f-11eb-3d26-ffab3de5f965
# ╟─4bcb1c1a-7762-11eb-1518-cf9f3d0e13dd
# ╟─6c83900a-76b2-11eb-30f1-07ef0afeaf98
# ╠═aaf3d4a8-76b2-11eb-001f-c563037c0b4c
# ╠═02e9348e-76a7-11eb-0749-b18ec7721cd3
# ╠═fe08c92e-76b6-11eb-1d71-e983816941c9
# ╠═850130fc-75ee-11eb-1363-67184e15f8ce
# ╠═ee915bfa-7670-11eb-27d8-7bc3ea3662ec
# ╠═415fc368-767a-11eb-0dcb-37b5fce5eb15
# ╠═03cbeb7a-7676-11eb-2c1e-2196b1ee104b
