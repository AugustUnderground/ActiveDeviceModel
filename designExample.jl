using Plots
using Flux
using Zygote
using CUDA
using BSON
using JLD2
using StatsBase
using NNlib
using FiniteDifferences
using DataFrames
using Optim
using LineSearches
using NumericIO
using Printf: @printf
using Chain
using ForwardDiff
using DataInterpolations

inspectdr();

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

boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);

normL(L) = (L - Lmin) / (Lmax - Lmin);
realL(L′) = L′ * (Lmax - Lmin) + Lmin;
    
normF(F) = (F - Fmin) / (Fmax - Fmin);
realF(F′) = F′ * (Fmax - Fmin) + Fmin;

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
    function operator(p1, id, p2, vds, vbs)
        #input = adjoint(hcat(p1, id, p2, vds, exp.(vds), vbs, sqrt.(abs.(vbs))));
        input = adjoint(hcat(p1, p2, id, vds, vbs));
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

opνn, νln = loadOperator("./model/current/L-vdsat-ptmn90.bson");
opνp, νlp = loadOperator("./model/current/L-vdsat-ptmp90.bson");
opγn, γln = loadOperator("./model/current/L-gmid-ptmn90.bson");
opγp, γlp = loadOperator("./model/current/L-gmid-ptmp90.bson");
opνn, νfn = loadOperator("./model/current/fug-vdsat-ptmn90.bson");
opνp, νfp = loadOperator("./model/current/fug-vdsat-ptmp90.bson");
opγn, γfn = loadOperator("./model/current/fug-gmid-ptmn90.bson");
opγp, γfp = loadOperator("./model/current/fug-gmid-ptmp90.bson");

opνn, νln = loadOperator("./model/current/L-vdsat-ptmn45.bson");
opνp, νlp = loadOperator("./model/current/L-vdsat-ptmp45.bson");
opγn, γln = loadOperator("./model/current/L-gmid-ptmn45.bson");
opγp, γlp = loadOperator("./model/current/L-gmid-ptmp45.bson");
opνn, νfn = loadOperator("./model/current/fug-vdsat-ptmn45.bson");
opνp, νfp = loadOperator("./model/current/fug-vdsat-ptmp45.bson");
opγn, γfn = loadOperator("./model/current/fug-gmid-ptmn45.bson");
opγp, γfp = loadOperator("./model/current/fug-gmid-ptmp45.bson");

VDD  = 1.2;
VSS  = 0.0;
Vicm = 0.6;
Vocm = 0.6;

############################### SYM ########################################

M = 4.0;
K = 0.5;
Iref = 10e-6;
CL = 10e-12;

#mp6 = νfp(2.122855e+08, 9.977337e-06, 0.2, Vocm, 0.0)

#function symamp(L12, L34, L78, L90, gmid12, vdsat34, vdsat78, vdsat90)
#function symamp(fug12, fug34, L78, L90, gmid12, vdsat34, vdsat78, vdsat90)
function symamp(fug12, fug34, fug78, L90, gmid12, vdsat34, vdsat78, vdsat90)
    I68 = ((K * M) / 2 ) * Iref
    #MN8 = νln(L78, I68, vdsat78, Vocm, 0.0);
    #gds8, Vgs8, Jd8 = Matrix(MN8[:,[:gds, :Vgs, :Jd]]);
    MN8 = νfn(fug78, I68, vdsat78, Vocm, 0.0)
    gds8, Vgs8, Jd8, L78 = Matrix(MN8[:,[:gds, :Vgs, :Jd, :L]])
    #W78 = 1.04 * (I68 / Jd8)
    W78 = I68 / Jd8
    Vd = Vgs8
    #MP6 = νlp(L34, I68, vdsat34, Vocm, 0.0);
    #gds6, Vgs6, Jd6 = Matrix(MP6[:,[:gds, :Vgs, :Jd]]);
    MP6 = νfp(fug34, I68, vdsat34, Vocm, 0.0)
    gds6, Vgs6, Jd6, L34 = Matrix(MP6[:,[:gds, :Vgs, :Jd, :L]])
    W56 = I68 / Jd6
    W34 = W56 / M
    W56 += 0.04W34
    Vb = Vc = (VDD - Vgs6)
    I12 = (K / 2) * Iref
    #Va = 0.06
    #Va = 0.10
    #Va = optimize( (Va) -> abs(γln(L12, I12, gmid12, (Vc - Va), -Va).Vgs[1] - (Vicm - Va))
    #              , VSS, VDD).minimizer
    #MN1 = γln(L12, I12, gmid12, (Vc - Va), -Va);
    #gm1, Jd1 = Matrix(MN1[:,[:gm, :Jd]]);
    Va = optimize( (Va) -> abs(γfn(fug12, I12, gmid12, (Vc - Va), -Va).Vgs[1] - (Vicm - Va))
                  , VSS, VDD).minimizer
    MN1 = γfn(fug12, I12, gmid12, (Vc - Va), -Va)
    gm1, Jd1, L12 = Matrix(MN1[:,[:gm, :Jd, :L]])
    W12 = I12 / Jd1
    I90 = K * Iref
    MN9 = νln(L90, I90, vdsat90, Va, 0.0);
    Vgs9, Jd9 = Matrix(MN9[:,[:Vgs, :Jd]]);
    #MN9 = νfn(fug90, I90, vdsat90, Va, 0.0)
    #Vgs9, Jd9, L90 = Matrix(MN9[:,[:Vgs, :Jd, :L]])
    W9 = I90 / Jd9
    W0 = W9 / K
    W9 += 0.04W0
    #W9 += 0.13W0
    rout = 1 / (gds6 + gds8)
    A₀ = M * gm1 * rout 
    A₀dB = 20 * log10(abs(A₀))
    f₀ = 1 / (2π * CL * rout)
    SR = I68 / CL
    return Dict( "A0dB" => A₀dB, "f0" =>  f₀, "SR" => SR
               , "gds6" => gds6, "gds8" => gds8, "gm1" => gm1
               , "L12" => L12, "L34" => L34, "L78" => L78, "L90" => L90
               , "W12" => W12, "W34" => W34, "W56" => W56, "W78" => W78
               , "W9" => W9, "W0" => W0 )
end;

gmid12 = 15.0;
vdsat12 = 0.1;
vdsat34 = 0.2;
vdsat78 = 0.2;
vdsat90 = 0.1;
cL90 = 8e-7;
#cL90 = 5e-7;

function gainTarget(X)
    #L12, L34, L78 = realL.(X);
    #design = symamp(L12, L34, L78, cL90, gmid12, vdsat34, vdsat78, vdsat90)
    fug12, fug34, fug78 = realF.(X)
    design = symamp(fug12, fug34, fug78, cL90, gmid12, vdsat34, vdsat78, vdsat90)
    #fug12, fug34, L78 = [realF(X[1]) realF(X[2]) realL(X[3])]
    #design = symamp(fug12, fug34, L78, cL90, gmid12, vdsat34, vdsat78, vdsat90)
    return sqrt(abs2( design["A0dB"] - A0Target ))
end;

function sizing(target)
    res = optimize(target, zeros(3), ones(3), rand(3), Fminbox(GradientDescent()))
    #L12, L34, L78 = realL.(res.minimizer)
    #design = symamp(L12, L34, L78, cL90, gmid12, vdsat34, vdsat78, vdsat90)
    fug12, fug34, fug78 = realF.(res.minimizer)
    design = symamp(fug12, fug34, fug78, cL90, gmid12, vdsat34, vdsat78, vdsat90)
    #fug12, fug34, L78 = [realF(res.minimizer[1]), realF(res.minimizer[2]), realL(res.minimizer[3])]
    #design = symamp(fug12, fug34, L78, cL90, gmid12, vdsat34, vdsat78, vdsat90)
    W12, W34, W56, W78, W9, W0 = [ design[d] for d = ["W12" "W34" "W56" "W78" "W9" "W0"]]
    L12, L34, L78, L90 = [ design[d] for d = ["L12" "L34" "L78" "L90"]]
    @printf("* A₀dB  =  %.4e\n", design["A0dB"])
    @printf("* f₀    =  %.4e\n", design["f0"])
    @printf("* SR    =  %.4e\n", design["SR"])
    @printf("* gds6  =  %.4e\n", design["gds6"])
    @printf("* gds8  =  %.4e\n", design["gds8"])
    @printf("* gm1   =  %.4e\n", design["gm1"])
    @printf("* fug12 =  %.4e\n", fug12)
    @printf("* fug34 =  %.4e\n", fug34)
    @printf("* fug78 =  %.4e\n", fug78)
    @printf(".param L12 =  %.4e\n", L12)
    @printf(".param L34 =  %.4e\n", L34)
    @printf(".param L78 =  %.4e\n", L78)
    @printf(".param L90 =  %.4e\n", L90)
    @printf(".param W12 =  %.4e\n", W12)
    @printf(".param W34 =  %.4e\n", W34)
    @printf(".param W56 =  %.4e\n", W56)
    @printf(".param W78 =  %.4e\n", W78)
    @printf(".param W9  =  %.4e\n", W9)
    @printf(".param W0  =  %.4e\n", W0)
    return nothing
end;

Lmin = 3.0e-7;
Lmax = 5.0e-6;
Linit = Lmin;
Fmin = 2.5e7;
Fmax = 1.0e9;
A0Target = 48.0;

@time sizing(gainTarget)


@time res = optimize( gainTarget, zeros(3), ones(3), rand(3), Fminbox(GradientDescent()))
design = symamp(realF.(res.minimizer)..., cL90, gmid12, vdsat34, vdsat78, vdsat90)

L12, L34, L78 = realL.(res.minimizer)
design = symamp(L12, L34, L78, L90, gmid12, vdsat34, vdsat78, vdsat90)


fug12, fug34, fug78, fug90 = realF.(rand(4))






########################################################################################

Ibias = 10e-6;
gmid1 = 15.0;
vdsat346 = 0.2;
vdsat578 = 0.1;
Cc = 3.0e-12;
#cL578 = 8e-7;
cL578 = 5e-7;
M = 5.0;


function moamp(fug12, fug346, L578, gmid1, vdsat346, vdsat578)
    I67 = Ibias * M
    MP6 = νfp(fug346, I67, vdsat346, (VDD - Vocm), 0.0)
    L346, gds6, gm6, Vgs6, Jd6 = Matrix(MP6[:,[:L, :gds, :gm, :Vgs, :Jd]])
    W6 = I67 / Jd6
    I1234 = Ibias / 2.0
    MP4 = νfp(fug346, I1234, vdsat346, (VDD - Vgs6), 0.0)
    gds4, Vgs4, Jd4 = Matrix(MP4[:,[:gds, :Vgs, :Jd]])
    W34 = I1234 / Jd4
    Vy = Vz = (VDD - Vgs4)
    Vx = optimize( (Vx) -> abs(γfn(fug12, I1234, gmid1, (Vy - Vx), -Vx).Vgs[1] - (Vicm - Vx))
                  , VSS, VDD).minimizer
    MN1 = γfn(fug12, I1234, gmid1, (Vy - Vx), -Vx)
    L12, gm1, gds1, Vgs1, Jd1 = Matrix(MN1[:,[:L, :gm, :gds, :Vgs, :Jd]])
    W12 = I1234 / Jd1
    MN5 = νln(L578, Ibias, vdsat578, Vx, 0.0)
    Vgs5, Jd5 = Matrix(MN5[:,[:Vgs, :Jd]])
    W8 = Ibias / Jd5
    W5 = 1.04W8
    W7 = W5 * M
    MN7 = νln(L578, I67, vdsat578, Vocm, 0.0)
    gds7, Vgs7 = Matrix(MN7[:,[:gds, :Vgs]])
    A₀ = (gm1 / (gds1 + gds4)) * (gm6 / (gds7 + gds6))
    A₀dB = 20 * log10( abs( A₀ ) )
    f₀ = ((gds1 + gds4) * (gds7 + gds6)) / (2π * Cc * gm6)
    SR = I67 / Cc
    return Dict( "A0dB" => A₀dB, "f0" =>  f₀, "SR" => SR
               , "gds6" => gds6, "gds7" => gds7, "gds1" => gds1, "gds4" => gds4
               , "gm1" => gm1, "gm6" => gm6
               , "L12" => L12, "L346" => L346, "L578" => L578
               , "W12" => W12, "W34" => W34, "W6" => W6
               , "W5" => W5, "W8" => W8, "W7" => W7 )
end;

function gainTarget(X)
    fug12, fug346 = realF.(X);
    design = moamp(fug12, fug346, cL578, gmid1, vdsat346, vdsat578)
    return sqrt(abs2( design["A0dB"] - A0Target))
end;

function sizing(target)
    res = optimize(target, zeros(2), ones(2), rand(2), Fminbox(GradientDescent()))
    fug12, fug346 = realF.(res.minimizer);
    #L12, L346, L578 = [8e-7 5e-7 1e-6 ]
    design = moamp(fug12, fug346, cL578, gmid1, vdsat346, vdsat578)
    W12, W34, W5, W8, W7, W6 = [ design[d] for d = ["W12" "W34" "W5" "W8" "W7" "W6"]]
    L12, L346, L578 = [ design[d] for d = ["L12" "L346" "L578"]]
    @printf("* A₀dB     =  %.4e\n", design["A0dB"])
    @printf("* f₀       =  %.4e\n", design["f0"])
    @printf("* SR       =  %.4e\n", design["SR"])
    @printf("* gds1     =  %.4e\n", design["gds1"])
    @printf("* gds4     =  %.4e\n", design["gds4"])
    @printf("* gds7     =  %.4e\n", design["gds7"])
    @printf("* gds6     =  %.4e\n", design["gds6"])
    @printf("* gm1      =  %.4e\n", design["gm1"])
    @printf("* gm6      =  %.4e\n", design["gm6"])
    @printf("* fug12    =  %.4e\n", fug12)
    @printf("* fug346   =  %.4e\n", fug346)
    @printf(".param L12     =  %.4e\n", L12)
    @printf(".param L346    =  %.4e\n", L346)
    @printf(".param L578    =  %.4e\n", L578)
    @printf(".param W12     =  %.4e\n", W12)
    @printf(".param W34     =  %.4e\n", W34)
    @printf(".param W5      =  %.4e\n", W5)
    @printf(".param W6      =  %.4e\n", W6)
    @printf(".param W7      =  %.4e\n", W7)
    @printf(".param W8      =  %.4e\n", W8)
    return nothing
end;

Lmin = 3.0e-7;
Lmax = 5.0e-6;
Linit = Lmin;
Fmin = 2.5e7;
Fmax = 1.0e9;
A0Target = 84.0;


@time nelist = sizing(gainTarget)

@time res = optimize(gainTarget, zeros(3), ones(3), rand(3), Fminbox(GradientDescent()))
L12, L346, L578 = realL.(res.minimizer)
design = moamp(L12, L346, L578, gmid1, vdsat4, vdsat5, vdsat6, vdsat7)



