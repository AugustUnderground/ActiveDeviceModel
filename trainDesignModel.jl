#####################
## Dependencies
#####################

using Dates
using Random
using Plots
using StatsBase
using DataFrames
using JLD2
using BSON
using CUDA
using MLDataUtils
using Flux
using Flux: @epochs
using Logging
using Printf: @printf
using NumericIO
using Chain

######################
## Setup
######################

# File System
paramName   = "L-vdsat";
deviceName  = "ptmp45";
deviceType  = :p;
timeStamp   = string(Dates.now());
modelDir    = "./model/$(paramName)-$(deviceName)-$(timeStamp)";
dataDir     = "../data";
mosFile     = "$(dataDir)/$(deviceName).jld";
modelFile   = "$(modelDir)/$(paramName)-$(deviceName).bson";
Base.Filesystem.mkdir(modelDir);

# Don't allow scalar operations on GPU
CUDA.allowscalar(false);

# Set Plot Backend
#unicodeplots();
inspectdr();

# The Seed of RNGesus
#rngSeed = deviceType == :n ? 666 : 999;
rngSeed = 666;

######################
## Data
######################

# Box-Cox ↔ Cox-Box Transformation
boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);

# Handle JLD2 dataframe
dataFrame = jldopen((f) -> f["database"], mosFile, "r");

# Processing, Fitlering, Sampling and Shuffling
dataFrame.QVgs  = dataFrame.Vgs.^2.0;
dataFrame.EVds  = ℯ.^(dataFrame.Vds);
dataFrame.RVbs  = abs.(dataFrame.Vbs).^0.5;
dataFrame.gmid  = dataFrame.gm ./ dataFrame.id;
dataFrame.A0    = dataFrame.gm ./ dataFrame.gds;
dataFrame.Jd    = dataFrame.id ./ dataFrame.W;
dataFrame.Veg   = dataFrame.Vgs .- dataFrame.vth;
dataFrame.Lgm   = log10.(dataFrame.gm);
dataFrame.Lgds  = log10.(dataFrame.gds);
dataFrame.Lid   = log10.(dataFrame.id);

mask = @chain dataFrame begin
            (isinf.(_) .| isnan.(_))
            Matrix(_)
            sum(_ ; dims = 2)
            (_ .== 0)
            vec()
            collect()
       end;
mdf = dataFrame[mask, : ];

### GOOD STUFF (L,vdsat) #####
paramsX = [ "L", "vdsat", "id", "Vds", "Vbs" ]; 
paramsY = [ "fug", "gmid", "gm", "gds", "Vgs", "Jd", "vth" ]; 
maskBCX = paramsX .∈([ "id" ],);
maskBCY = paramsY .∈([ "fug", "gmid", "gm", "gds", "Jd" ],);
###################

### GOOD STUFF (L,gmid) #####
#paramsX = [ "L", "gmid", "id", "Vds", "Vbs" ]; 
#paramsY = [ "fug", "vdsat", "gm", "gds", "Vgs", "Jd", "vth" ]; 
#maskBCX = paramsX .∈([ "gmid", "id" ],);
#maskBCY = paramsY .∈([ "fug", "gm", "gds", "Jd" ],);
###################

### GOOD STUFF (fug,vdsat) #####
#paramsX = [ "fug", "vdsat", "id", "Vds", "Vbs" ]; 
#paramsY = [ "L", "gmid", "gm", "gds", "Vgs", "Jd", "vth" ]; 
#maskBCX = paramsX .∈([ "fug", "id" ],);
#maskBCY = paramsY .∈([ "gm", "gmid", "gds", "Jd" ],);
###################

### GOOD STUFF (fug,gmid) #####
#paramsX = [ "fug", "gmid", "id", "Vds", "Vbs" ]; 
#paramsY = [ "L", "vdsat", "gm", "gds", "Vgs", "Jd", "vth" ]; 
#maskBCX = paramsX .∈([ "fug", "gmid", "id" ],);
#maskBCY = paramsY .∈([ "gm", "gds", "Jd" ],);
###################

# Number of In- and Outputs, for respective NN Layers
numX = length(paramsX);
numY = length(paramsY);

## Sample Data for appropriate distribution

# Random Uniform with Probability Weights on gds
#df = mdf[ StatsBase.sample( MersenneTwister(rngSeed)
#                          , 1:size(mdf, 1)
#                          , StatsBase.pweights(mdf.gds)
#                          , 4000000
#                          ; replace = false
#                          , ordered = false )
#        , : ];

# When VGS > Vth and VDS ≥ (VGS – Vth):
satMask = ((mdf.Vds .>= (mdf.Vgs .- mdf.vth)) .& (mdf.Vgs .> mdf.vth));

# Sample 3/4ths of Data in Saturation Region with probability weighted Id
#sdf = mdf[ ifelse( deviceType == :n
#                 , mdf.Vds .>= (mdf.Vgs .- mdf.vth) 
#                 , mdf.Vds .<= (mdf.Vgs .+ mdf.vth) ) 
#         , :];
sdf = mdf[satMask, : ];
sSamp = sdf[ StatsBase.sample( MersenneTwister(rngSeed)
                             , 1:size(sdf, 1)
                             , StatsBase.pweights(sdf.gds)
                             , 3000000
                             ; replace = false
                             , ordered = false )
           , : ];
# Sample 1/3rd of Data in Triode Region without weights
#tdf = mdf[ ifelse( deviceType == :n
#                 , mdf.Vds .<= (mdf.Vgs .- mdf.vth) 
#                 , mdf.Vds .>= (mdf.Vgs .+ mdf.vth) ) 
#         , :];
#tdf = mdf[((mdf.Vds .>= (mdf.Vgs .- mdf.vth)) .& (mdf.Vgs .> mdf.vth)), : ];
tdf = mdf[.!satMask, : ];
tSamp = tdf[ StatsBase.sample( MersenneTwister(rngSeed)
                             , 1:size(tdf, 1)
                             , StatsBase.pweights(tdf.gds)
                             , 1000000
                             ; replace = false
                             , ordered = false )
           , : ];
# Join samples and shuffle all observations
df = shuffleobs(vcat(tSamp, sSamp));

# Convert to Matrix for Flux
rawX = Matrix(df[ : , paramsX ])';
rawY = Matrix(df[ : , paramsY ])';

# Transform according to mask
λ = 0.2;
rawX[maskBCX,:] = boxCox.(abs.(rawX[maskBCX,:]); λ = λ);
rawY[maskBCY,:] = boxCox.(abs.(rawY[maskBCY,:]); λ = λ);

# Rescale data to [0;1]
utX = StatsBase.fit(UnitRangeTransform, rawX; dims = 2, unit = true); 
utY = StatsBase.fit(UnitRangeTransform, rawY; dims = 2, unit = true); 

dataX = StatsBase.transform(utX, rawX);
dataY = StatsBase.transform(utY, rawY);

# Split data in training and validation set
splitRatio = 0.8;
trainX,validX = splitobs(dataX, splitRatio);
trainY,validY = splitobs(dataY, splitRatio);

# Create training and validation Batches
batchSize = 2000;
trainSet = Flux.Data.DataLoader( (trainX, trainY)
                               , batchsize = batchSize
                               , shuffle = true );
validSet = Flux.Data.DataLoader( (validX, validY)
                               , batchsize = batchSize
                               , shuffle = true );

######################
## Model
######################

# Neural Network Architecture
γ = Flux.Chain( Flux.Dense(numX,  128,    Flux.relu)
              , Flux.Dense(128,   256,    Flux.relu) 
              , Flux.Dense(256,   512,    Flux.relu) 
              , Flux.Dense(512,   1024,   Flux.relu) 
              , Flux.Dense(1024,  512,    Flux.relu) 
              , Flux.Dense(512,   256,    Flux.relu) 
              , Flux.Dense(256,   128,    Flux.relu) 
              , Flux.Dense(128,   numY,   Flux.relu) 
              ) |> gpu;

# Optimizer Parameters
η   = 0.001;
β₁  = 0.9;
β₂  = 0.999;

# ADAM Optimizer
optim = Flux.Optimise.ADAM(η, (β₁, β₂)) |> gpu;

######################
## Training
######################

# Loss/Objective/Cost function for Training and Validation
mse(x, y) = Flux.Losses.mse(γ(x), y, agg = mean);
mae(x, y) = Flux.Losses.mae(γ(x), y, agg = mean);
hub(x, y) = Flux.Losses.huber_loss(γ(x), y, δ = 1, agg = mean);

# Model Parameters (Weights)
θ = Flux.params(γ) |> gpu;

# Training Loop
function trainModel()
    trainMSE = map(trainSet) do batch               # iterate over batches in train set
        gpuBatch = batch |> gpu;                    # make sure batch is on GPU
        error,back = Flux.Zygote.pullback(() -> mse(gpuBatch...), θ);
        ∇ = back(one(error |> cpu)) |> gpu;         # gradient based on error
        Flux.update!(optim, θ, ∇);                  # update weights
        return error;                               # return MSE
    end;
    validMAE = map(validSet) do batch               # no gradients required
        gpuBatch = batch |> gpu;
        error,back = Flux.Zygote.pullback(() -> mae(gpuBatch...), θ);
        return error;                               # iterate over validation data set
    end;
    meanMSE = mean(trainMSE);                       # get mean training error over epoch
    meanMAE = mean(validMAE);                       # get mean validation error over epoch
    @printf( "[%s] MSE = %s and MAE = %s\n"
           , Dates.format(now(), "HH:MM:SS")
           , formatted(meanMSE, :ENG, ndigits = 4) 
           , formatted(meanMAE, :ENG, ndigits = 4) )
    if meanMAE < lowestMAE                          # if model has improved
        bson( modelFile                             # save the current model (cpu)
            , name = deviceName
            , type = deviceType
            , parameter = paramName
            , model = (γ |> cpu) 
            , paramsX = paramsX
            , paramsY = paramsY
            , utX = utX
            , utY = utY 
            , maskX = maskBCX
            , maskY = maskBCY
            , lambda = λ );
        global lowestMAE = meanMAE;                 # update previous lowest MAE
        @printf( "\tNew Model Saved with MAE: %s\n" 
               , formatted(meanMAE, :ENG, ndigits = 4) )
    end
    return [meanMSE meanMAE];                       # mean of error for all batches
end

### Run Training
numEpochs = 50;                                    # total number of epochs
lowestMAE = Inf;                                    # initialize MAE with ∞
errs = [];                                          # Array of training and validation losses

@epochs numEpochs push!(errs, trainModel())         # Run Training Loop for #epochs

######################
## Evaluation
######################

# Reshape errors for a nice plot
losses = hcat( map((e) -> Float64(e[1]), errs)
             , map((e) -> Float64(e[2]), errs) );

# Plot Training Process
plot( 1:numEpochs, losses; lab = ["MSE" "MAE"]
    , xaxis = ("# Epoch", (1,numEpochs))
    , yaxis = ("Error", (0.0, ceil( max(losses...)
                                  , digits = 3 )) )
    , w = 2 )

## Use Current model ##
#γ = γ |> cpu;

## Load specific model ##
#modelFile = "./model/vdsat-ptmn90-2021-02-22T14:14:31.445/ptmn90.bson"
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
device  = model[:name];

## Reload DB ##########
#dataFrame = jldopen("../data/$(device).jld", "r") do file
#    file["database"];
#end;
#dataFrame.QVgs = dataFrame.Vgs.^2.0;
#dataFrame.EVds = exp.(dataFrame.Vds);
#dataFrame.RVbs = sqrt.(abs.(dataFrame.Vbs));
#dataFrame.gmid = dataFrame.gm ./ dataFrame.id;
#dataFrame.A0 = dataFrame.gm ./ dataFrame.gds;
#dataFrame.Jd = dataFrame.id ./ dataFrame.W;
#msk = @chain dataFrame begin
#            (isinf.(_) .| isnan.(_))
#            Matrix(_)
#            sum(_ ; dims = 2)
#            (_ .== 0)
#            vec()
#            collect()
#      end;
#mdf = dataFrame[msk, : ];
#boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
#coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);
#inspectdr();
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

L = rand(filter(l -> l < 1.0e-6, unique(mdf.L)));
W = rand(filter(w -> w < 2.0e-6, unique(mdf.W)));

fugs =  exp10.(range(8, stop = 10, length = 10));

Id = 10e-6;
Vbs = 0.0;
Vds = 0.6;

param1,param2 = split(param, "-");

traceT = sort( mdf[ ( (mdf[:,"L"] .== L)
                   .& (mdf[:,"W"] .== W)
                   .& (mdf[:,"Vbs"] .== Vbs)
                   .& (mdf[:,"Vds"] .== Vds) )
                  , : ]
             , [param2] );

len = length(traceT[:,param2]);

p1 = traceT[:,param1];
p2 = traceT[:,param2];
#id = fill(Id, len);
id = traceT.id;
lid = log10.(id);
vds = traceT[:,"Vds"];
evds = exp.(vds);
vbs = traceT[:,"Vbs"];
rvbs = sqrt.(abs.(vbs));

x = [ p1 p2 id vds vbs ]';
y = predict(x);

plot( p2, y.gds; yscale = :log10
    , lab = "Aprx", w = 2, xaxis = param);
plot!(p2, traceT.gds; yscale = :log10, lab = "True", w = 2)

plot( para, y.A0; yscale = :log10
    , lab = "Aprx", w = 2, xaxis = param, yaxis = "gds" );
#plot!(para, y.gm ./ y.gds; lab = "Aprx", w = 2)
plot!(para, traceT.A0; lab = "True", w = 2)
#plot!(para, traceT.gm ./ traceT.gds; lab = "True", w = 2)

plot(para, y.gm; lab = "Aprx", w = 2);
plot!(para, traceT.gm; lab = "True", w = 2)

plot(para, y.gds; lab = "Aprx", w = 2);
plot!(para, traceT.gds; lab = "True", w = 2)

#plot!(para, y.gm ./ y.gds; lab = "Aprx", w = 2);
#plot!(para, traceT.gm ./ traceT.gds; lab = "True", w = 2);
#plot!(para, y.A0; lab = "Aprx", w = 2);
#plot!(para, traceT.A0; lab = "True", w = 2)


traceO = sort( mdf[ ( (mdf[:,"L"] .== L)
                   .& (mdf[:,"W"] .== W)
                   .& (mdf[:,"Vbs"] .== Vbs)
                   .& (mdf[:,"Vgs"] .== 0.6) )
                  , : ]
             , ["Vds" ] );


