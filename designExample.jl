using Plots
using PlutoUI
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

opνn90, νln90 = loadOperator("./model/current/l-vdsat-nmos90.bson");
opνp90, νlp90 = loadOperator("./model/current/l-vdsat-pmos90.bson");
opγn90, γln90 = loadOperator("./model/current/l-gmid-nmos90.bson");
opγp90, γlp90 = loadOperator("./model/current/l-gmid-pmos90.bson");

VDD  = 1.2;
VSS  = 0.0;
Vicm = 0.5;
Vocm = 0.6;
Lmin = 1.0e-7;
Lmax = 1.0e-6;
Linit = 3Lmin;

M = 5.0;
K = 0.5;
Iref = 50e-6;
CL = 10e-12;

L90 = 3e-7;

function symamp(L12, L34, L78) # vdsat12, vdsat34, vdsat78, vdsat90
    I68 = ((K * M) / 2 ) * Iref;
    MN8 = νln90(L78, I68, vdsat78, Vocm, 0.0);
    W78, gds8, Vgs8, Jd8 = Matrix(MN8[:,[:W, :gds, :Vgs, :Jd]]);
    W78 = I68 / Jd8;
    Vd = Vgs8;
    MP6 = νlp90(L34, I68, vdsat34, Vocm, 0.0);
    W56, gds6, Vgs6, Jd6 = Matrix(MP6[:,[:W, :gds, :Vgs, :Jd]]);
    W56 = I68 / Jd6;
    W34 = W56 / M;
    Vb = Vc = (VDD - Vgs6);
    I12 = (K / 2) * Iref;
    Va = Vc / 2;
    #MN1 = γln90(L12, I12, gmid12, (Vb - Va), -Va);
    #MN1 = νln90(L12, I12, vdsat12, VDD/3, 0.0);
    MN1 = γln90(L12, I12, gmid12, Va, 0.0);
    W12, gm1, Jd1 = Matrix(MN1[:,[:W, :gm, :Jd]]);
    W12 = I12 / Jd1;
    I90 = K * Iref;
    MN9 = νln90(L90, I90, vdsat90, Va, 0.0);
    W9, Vgs9, Jd9 = Matrix(MN9[:,[:W, :Vgs, :Jd]]);
    W9 = I90 / Jd9;
    W0 = W9 / K;
    rout = 1 / (gds6 + gds8);
    A₀ = M * gm1 * rout ;
    A₀dB = 20 * log10(abs(A₀));
    #ω₀ = 1 / (CL * rout);
    f₀ = 1 / (2π * CL * rout);
    SR = I68 / CL;
    return [A₀dB, f₀, SR, W12, W34, W56, W78, W9, W0]
end;

gmid12 = 12;
vdsat34 = 0.2;
vdsat78 = 0.2;
vdsat90 = 0.2;

A₀Target = 45.0;
f₀Target = 150e3;
 
lowerBound = [0.0, 0.0, 0.0, 0.0];
upperBound = [1.0, 1.0, 1.0, 1.0];
gInitial = [0.4, 0.6, 0.5, 0.2];
fInitial = [0.2, 0.2, 0.2, 0.2];
   
#lower   = [0.1, 0.1, 0.1, 0.1, 2e-7, 2e-7, 2e-7, 2e-7];
#upper   = [0.3, 0.3, 0.3, 0.3, 2e-6, 2e-6, 2e-6, 2e-6];
#init    = [0.1, 0.1, 0.1, 0.1, 3e-7, 3e-7, 3e-7, 3e-7];
#init = rand(8);
#norm(X) = (X - lower) ./ (upper - lower);
#mron(X′) = X′ .* (upper - lower) + lower;

#optimAlgorithm = ConjugateGradient();
optimAlgorithm = GradientDescent();
#optimAlgorithm = NelderMead();
#optimAlgorithm = GradientDescent(linesearch=LineSearches.BackTracking(order=2));
#optimAlgorithm = LBFGS();

optimOptions = Optim.Options( g_tol = 1e-3
                            , x_tol = 1e-3
                            , f_tol = 1e-3
                            , time_limit = 25.0 
                            , show_trace = true
                            ); 

function symampGainObjective(X)
    A₀dB, _ = symamp(realL.(X)...);
    return abs2(A₀dB - A₀Target)
end;

a0Results = optimize( symampGainObjective, zeros(3), ones(3), rand(3)
                    , Fminbox(optimAlgorithm), optimOptions)

designParamters = realL.(a0Results.minimizer)
design = symamp(designParamters...)





function symampBWObjective(X)
    _, ω₀, _ = symamp(realL.(X)...);
    return -(ω₀ / 2π)
end;

f0Results = optimize( symampBWObjective, norm(lower), norm(upper), rand(8)
                    , Fminbox(optimAlgorithm), optimOptions)

designParamters = mron(f0Results.minimizer)
design = symamp(designParamters...)

