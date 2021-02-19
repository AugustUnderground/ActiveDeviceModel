using Plots
using Statistics
using PlutoUI
using Flux
using Zygote
using CUDA
using BSON
using PyCall
using ScikitLearn
using NNlib
using Optim
using NLsolve

using JLD2
using DataFrames
using StatsBase

using OhMyREPL

using DiffEqFlux
using ForwardDiff
using FiniteDiff

joblib = pyimport("joblib");
#pyplot();
unicodeplots();

modelPathₙ = "./model/ptmn90-2021-01-19T16:52:09.8/ptmn90";
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
    return Float64.( ((length(size(X)) < 2) ? [X'] : X') |>
                     trafoXₙ.transform |> 
                     adjoint |> φₙ |> adjoint |>
                     trafoYₙ.inverse_transform |> 
                     adjoint )
end;

function nmos(Vgs, Vds, W, L)
    return predictₙ([ Vgs' ; (Vgs.^2)' 
                    ; Vds' ; exp.(Vds)'
                    ; W' ; L' ])
end;

function nmos_id(Vgs, Vds, W, L)
    prd = nmos(Vgs, Vds, W, L)
    return prd[ first(indexin(["id"], paramsYₙ)), : ][1];
end;


vgs = collect(0.0:0.01:1.2);
qvgs = vgs.^2.0;
vds = collect(0.0:0.01:1.2);
evds = exp.(vds);
len = length(vgs);
W = 1.0e-6;
w = fill(W, len);
L = 3.0e-7;
l = fill(L, len);
vg = 0.6;
vgc = fill(vg, len);
qvgc = vgc.^2.0;
vd = 0.6;
vdc = fill(vd, len);
evdc = exp.(vdc);
vbc = zeros(len);

nmos(vgs, vdc, w, l)

nmos_id(vgs, vdc, w, l)

xt = [ vgs vdc  w l ]

[ gradient(nmos_id, x...)  for x in eachrow(xt) ]







ncm = jldopen("../data/cm-nxh035.jld", "r") do file
    file["database"];
end;

ncm.M0_vds = round.(ncm.M0_vds; digits = 3);
ncm.O = round.(ncm.O; digits = 3);

res = ncm[ ( (ncm.Mcm11 .== ncm.Mcm12)
          .& (ncm.M0_vds .== ncm.O) )
         , ["Mcm11", "Mcm12", "M0_vds", "O", "M0_id", "M1_id"] ];









################################################################################
#
modelPathₚ = "./model/ptmp90-2021-01-13T12:04:05.819/ptmp90";
modelFileₚ = modelPathₚ * ".bson";
trafoInFileₚ = modelPathₚ * ".input";
trafoOutFileₚ = modelPathₚ * ".output";
modelₚ = BSON.load(modelFileₚ);
φₚ = modelₚ[:model];
trafoXₚ = joblib.load(trafoInFileₚ);
trafoYₚ = joblib.load(trafoOutFileₚ);
paramsXₚ = modelₚ[:paramsX];
paramsYₚ = modelₚ[:paramsY];

function predictₚ(X)
    return Float64.(((length(size(X)) < 2) ? [X'] : X') |>
         trafoXₚ.transform |> 
         adjoint |> φₚ |> adjoint |>
         trafoYₚ.inverse_transform |> 
         adjoint )
end;

function pmos(Vgs, Vds, W, L)
    return predictₚ([Vds ; Vds.^2 ; Vgs ; exp.(Vgs); W ; L])
end

ϕDD = 1.2;
ϕSS = 0.0;
ϕI1 = 0.7;
ϕI2 = 0.7;

ib = 50e-6;
    
W₁₂ = 2e-6;
W₃₄ = 4e-6;
W₅₆ = 4e-6;
W₇₈ = 3e-6;
W₉₀ = 3e-6;

Lₘᵢₙ = 3e-7;

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
end

ΔKCL = (x) -> (1 / abs(sum(kcl(x))))

ϕ₀ = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

#jacobian(central_fdm(5,1), ΔKCL, ϕ₀) |> first
ΔKCL = grad(central_fdm(5,1), ΔKCL, ϕ₀) |> first


x = [ 0.0:0.01:2π |> collect |> adjoint 
    ; 0.0:0.01:2π |> collect |> adjoint ]

f = (x) -> sin(x[1]) + cos(x[2])

Δf = (x) -> central_fdm(5,1)(f, x)

∇f = (x) -> grad(central_fdm(5,1), f, x) |> first

y = map(f, eachcol(x));
∇y = hcat(map(∇f, eachcol(x))...);
plot(x[1,:], y);
plot!(x[1,:], ∇y[1,:]);
plot!(x[1,:], ∇y[2,:])

plot(x, f.(x));
plot!(x, Δf.(x));
plot!(x, ∇f.(x))

ncmPath = "../data/cm-nxh035.jld";

dataFrame = jldopen((f) -> f["database"], ncmPath, "r");














####################################################
# DESIGN MODEL EVALUATION
####################################################

modelFile = "./model/ptmn90-vdsat_Vds-2021-02-15T18:35:59.555/ptmn90.bson"
model   = BSON.load(modelFile);
γ       = model[:model];
paramsX = model[:paramsX];
paramsY = model[:paramsY];
utX     = model[:utX];
utY     = model[:utY];
maskBCX = model[:maskX];
maskBCY = model[:maskY];
λ       = model[:lambda];
param   = model[:parameter];
term    = model[:terminal];
device  = model[:name];

## Reload DB ##########
inspectdr();
dataFrame = jldopen("../data/$(device).jld", "r") do file
    file["database"];
end;
dataFrame.QVgs = dataFrame.Vgs.^2.0;
dataFrame.EVds = exp.(dataFrame.Vds);
dataFrame.RVbs = sqrt.(abs.(dataFrame.Vbs));
dataFrame.gmid = dataFrame.gm ./ dataFrame.id;
dataFrame.A0 = dataFrame.gm ./ dataFrame.gds;
dataFrame.Jd = dataFrame.id ./ dataFrame.W;
msk = @chain dataFrame begin
            (isinf.(_) .| isnan.(_))
            Matrix(_)
            sum(_ ; dims = 2)
            (_ .== 0)
            vec()
            collect()
      end;
mdf = dataFrame[msk, : ];

boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);
########################

mdf.Vgs = round.(mdf.Vgs, digits = 2);
mdf.Vds = round.(mdf.Vds, digits = 2);
mdf.Vbs = round.(mdf.Vbs, digits = 2);

### γ evaluation function for prediction characteristics

function predict(X)
    X[maskBCX,:] = boxCox.(abs.(X[maskBCX,:]); λ = λ);
    X′ = StatsBase.transform(utX, X);
    Y′ = γ(X′);
    Y = StatsBase.reconstruct(utY, Y′);
    Y[maskBCY,:] = coxBox.(Y[maskBCY,:]; λ = λ);
    return DataFrame(Float64.(Y'), String.(paramsY))
end;

L = 300e-9;
W = 2.0e-6
Id = 50e-6;
vdsat = 0.2;
Vbs = 0.0;
Vgs = 0.6

function loss(Vds, vdsat)
    x = [ L Id vdsat Vds (ℯ ^ Vds) Vbs (Vbs ^ 0.5) ]';
    y = predict(x);
    ΔVg = abs(y.Vgs[1] - Vgs);
    ΔW  = abs((y.W[1] * 10^6) - (W * 10^6));
    ΔJ  = abs(y.Jd[1] - (Id / W))
    print("ΔVgs = $(lossVg) ; ΔW = $(lossW)\n")
    return (lossVg + lossW)
end;

opt = optimize((x) -> loss(x[1], 0.2), [0.6], Newton())
opt = optimize(f, lower, upper, initial_x, Fminbox(inner_optimizer); autodiff = :forward)
Vds = opt.minimizer[1]

x = [ L Id vdsat Vds (Vds ^ 2.0) Vbs (Vbs ^ 0.5) ]';
y = predict(x)




f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
inner_optimizer = GradientDescent()
results = optimize(f, lower, upper, initial_x, Fminbox(inner_optimizer); autodiff = :forward)
