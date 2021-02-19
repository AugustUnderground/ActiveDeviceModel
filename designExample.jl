using FiniteDifferences
using mna
using BSON

## Setup

# Box-Cox Transformation Utility Functions
boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);

# MOSFET Model Definition
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

# Load Trained Transistor Models
nmos90file = BSON.load("../model/ptmn90-2021-02-12T17:35:43.636/ptmn90.bson");
pmos90file = BSON.load("../model/ptmp90-2021-02-12T18:04:33.896/ptmp90.bson");

# Create Device-Structs
ptmn90 = MOSFET( nmos90file[:model]
               , String.(nmos90file[:paramsX])
               , String.(nmos90file[:paramsY])
               , nmos90file[:utX]
               , nmos90file[:utY]
               , nmos90file[:maskX]
               , nmos90file[:maskY]
               , nmos90file[:lambda] );

ptmp90 = MOSFET( pmos90file[:model]
               , String.(pmos90file[:paramsX])
               , String.(pmos90file[:paramsY])
               , pmos90file[:utX]
               , pmos90file[:utY]
               , pmos90file[:maskX]
               , pmos90file[:maskY]
               , pmos90file[:lambda] );

# Define Convenience Functions to evaluate Operating point
function nmos(Xd)
    X = Matrix(Xd[:,ptmn90.paramsX])';
    X[ptmn90.maskX,:] = boxCox.(abs.(X[ptmn90.maskX,:]); λ = ptmn90.lambda);
    X′ = StatsBase.transform(ptmn90.utX, X);
    Y′ = ptmn90.model(X′);
    Y = StatsBase.reconstruct(ptmn90.utY, Y′);
    Y[ptmn90.maskY,:] = coxBox.(Y[ptmn90.maskY,:]; λ = ptmn90.lambda);
    return DataFrame(Float64.(Y'), String.(ptmn90.paramsY))
end;

function pmos(Xd)
    X = Matrix(Xd[:,ptmp90.paramsX])';
    X[ptmp90.maskX,:] = boxCox.(abs.(X[ptmp90.maskX,:]); λ = ptmp90.lambda);
    X′ = StatsBase.transform(ptmp90.utX, X);
    Y′ = ptmp90.model(X′);
    Y = StatsBase.reconstruct(ptmp90.utY, Y′);
    Y[ptmp90.maskY,:] = coxBox.(Y[ptmp90.maskY,:]; λ = ptmp90.lambda);
    return DataFrame(Float64.(Y'), String.(ptmp90.paramsY))
end;

function mosfetOP(mos, Vgs, Vds, Vbs, W, L)
    valuesX = [W, L, Vgs, Vgs^2.0, Vds, exp(Vds), Vbs, sqrt(abs(Vbs))];
    input = DataFrame(Dict(ptmn90.paramsX .=> valuesX));
    result = mos == :n ? nmos(input) : pmos(input);
    return result
end;

function mosfetOP(Vgs, Vds, Vbs, W, L)
    paramsX = [ "W", "L", "Vgs", "QVgs", "Vds", "EVds", "Vbs", "RVbs"];
    valuesX = hcat(W, L, Vgs, Vgs.^2.0, Vds, exp.(Vds), Vbs, sqrt.(abs.(Vbs)));
    DataFrame(valuesX, paramsX);
end;

# Convert Operating Point Parameters to Small Signal Netlist
function op2nl(idx, OP, φG, φD, φS, φB)
    Dict([ "cgd$(idx)" => (:C, φG, φD, OPdf.cgd)
         , "cgs$(idx)" => (:C, φG, φS, OPdf.cgs)
         , "cgb$(idx)" => (:C, φG, φB, OPdf.cgb)
         , "cds$(idx)" => (:C, φD, φS, OPdf.cds)
         , "cdb$(idx)" => (:C, φD, φB, OPdf.cdb)
         , "csb$(idx)" => (:C, φS, φB, OPdf.csb)
         , "gm$(idx)"  => (:VCCS, φG, φS, φD, φS, OPdf.gm)
         , "gmb$(idx)" => (:VCCS, φB, φS, φD, φS, OPdf.gmb)
         , "gds$(idx)" => (:G, φD, φS, OPdf.gds) ]);
end;

## Single Ended, Two Stage OTA

# Nets
φSS = 0;
φIP = 1;
φIN = 2;
φI  = 3;
φZ  = 4;
φX  = 5;
φY  = 6;
φO  = 7;
φDD = 8;

# Prior Knowledge / Specifications
Vdd     = 1.2;
Vss     = 0.0;
Ibias   = 30e-6;
Vicm    = 0.6;
Vocm    = 0.6;
CL      = 10e-12;

# Testbench for Small-Signal Analysis
testbench = Dict([ "VDD" => (:V, φDD, φSS, Vdd)
                 , "VIP" => (:V, φIP, φSS, Vicm)
                 , "VIN" => (:V, φIN, φSS, Vicm)
                 , "VO"  => (:V, φO, φSS, Vocm)
                 , "CL"  => (:C, φO, φSS, CL) ]);

# Design Parameters
L12     = 300e-9;
L346    = 300e-9;
L578    = 300e-9;
vdsat12 = 0.2;
vdsat346= 0.2;
vdsat578= 0.2;

# Find Operating Point based on Prior Knowledge

# M6
# M7, (M5, M8)
# M3, M4
# M1, M2

# Get Operating Point Parameters
OP12 = mosfetOP(:n, (Vicm - Vz), (Vx - Vz), -Vz,  L12, W12)
OP34 = mosfetOP(:p, (Vdd - Vx), (Vdd - Vx), 0.0, L346, W34);
OP5  = mosfetOP(:n, Vi, Vz, 0.0, L578, W58);
OP8  = mosfetOP(:n, Vi, Vi, 0.0, L578, W58);
OP7  = mosfetOP(:n, Vi, Vocm, 0.0, L578, W7);
OP6  = mosfetOP(:p, (Vdd - Vx), (Vdd - Vocm), 0.0, L578, W7);

# Create Small-Signal Netlist
OTA = merge([ op2nl(1, OP12, φIN, φY, φZ, φSS) 
            , op2nl(2, OP12, φIP, φX, φZ, φSS)
            , op2nl(3, OP34, φY, φY, φDD, φDD)
            , op2nl(4, OP34, φY, φX, φDD, φDD)
            , op2nl(5, OP5, φI, φZ, φSS, φSS)
            , op2nl(7, OP7, φI, φO, φSS, φSS)
            , op2nl(8, OP8, φI, φI, φSS, φSS)
            , op2nl(6, OP6, φX, φO, φDD, φDD)
            ]... );

# Merge with Testbench
ckt = merge(OTA, testbench);

# Small-Signal Analysis
x = mna.analyze(ckt)
tf = transferFunction(x);
