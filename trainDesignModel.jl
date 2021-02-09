#####################
## Dependencies
#####################

using Dates
using Plots
using StatsBase
using Random
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

######################
## Setup
######################

# File System
timeStamp = string(Dates.now());
modelDir = "./model/des-" * timeStamp * "/";
dataDir = "../data/";
deviceName = "ptmn90";
mosFile = dataDir * deviceName * ".jld";
modelFile = modelDir * deviceName * ".bson";
trafoXFile = modelDir * deviceName * ".input";
trafoYFile = modelDir * deviceName * ".output";
Base.Filesystem.mkdir(modelDir);

# Don't allow scalar operations on GPU
CUDA.allowscalar(false);

# Set Plot Backend
inspectdr();
#unicodeplots();

rngSeed = 666;

######################
## Data
######################

# Handle JLD2 dataframe
dataFrame = jldopen(mosFile, "r") do file
    file["database"];
end;

# Processing, Fitlering, Sampling and Shuffling
dataFrame.QVgs = dataFrame.Vgs.^2.0;
dataFrame.EVds = exp.(dataFrame.Vds);
dataFrame.Vgs = round.(dataFrame.Vgs, digits = 2);
dataFrame.gmid = dataFrame.gm ./ dataFrame.id;
dataFrame.A0 = dataFrame.gm ./ dataFrame.gds;
dataFrame.idW = dataFrame.id ./ dataFrame.W;
#dataFrame.A0Log =  log10.(dataFrame.gm ./ dataFrame.gds);
#dataFrame.idWLog = log10.(dataFrame.id ./ dataFrame.W);
#dataFrame.fugLog = log10.(dataFrame.fug);

mask = (vec ∘ collect)(sum(Matrix(isinf.(dataFrame) .| isnan.(dataFrame)), dims = 2) .== 0);
dff = dataFrame[mask, : ];

numSamples = 666666;
idxSamples = StatsBase.sample( MersenneTwister(rngSeed)
                             , 1:(dff |> size |> first)
                             , StatsBase.pweights(dff.id)
                             , numSamples
                             ; replace = false
                             , ordered = false );

df = dff[idxSamples, :];

# Use all Parameters for training

# [id, vdsat, l, vds, vbs ] => [ id/w, vgs, gm, gds, fug]
paramsX = [ "L", "id", "vdsat", "Vds", "EVds"]; 
paramsY = [ "idW", "gm", "fug", "gds", "Vgs"]; #, "QVgs" ];

#paramsY = [ "Vgs", "idW", "A0" ] #, "fug" ];
#paramsX = [ "W", "L", "Vgs", "QVgs", "Vds", "EVds"];

# Number of In- and Outputs, for respective NN Layers
numX = length(paramsX);
numY = length(paramsY);

# Convert to Matrix for Flux
rawX = Matrix(df[:, paramsX ])';
rawY = Matrix(df[:, paramsY ])';

#boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
#coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);

#λ = 0.2;
#boxY = boxCox.(rawY; λ = λ);

utX = StatsBase.fit(UnitRangeTransform, rawX; dims = 2, unit = true); 
utY = StatsBase.fit(UnitRangeTransform, rawY; dims = 2, unit = true); 
#utY = StatsBase.fit(UnitRangeTransform, boxY; dims = 2, unit = true); 

dataX = StatsBase.transform(utX, rawX);
dataY = StatsBase.transform(utY, rawY);
#dataY = StatsBase.transform(utY, boxY);

# Split data in training and validation set
splitRatio = 0.8;
trainX,validX = splitobs(dataX, splitRatio);
trainY,validY = splitobs(dataY, splitRatio);

# Create training and validation Batches
batchSize = 666;
trainSet = Flux.Data.DataLoader( (trainX, trainY)
                               , batchsize = batchSize
                               , shuffle = true );
validSet = Flux.Data.DataLoader( (validX, validY)
                               , batchsize = batchSize
                               , shuffle = true );

######################
## Model
######################

# Neural Network
γ = Chain( Dense(numX, 128, relu)
         , Dense(128, 256, relu) 
         , Dense(256, 512, relu) 
         , Dense(512, 1024, relu) 
         , Dense(1024, 512, relu) 
         , Dense(512, 256, relu) 
         , Dense(256, 128, relu) 
         , Dense(128, numY, relu) 
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
        bson( modelFile
            , model = (γ |> cpu)
            , paramsX = paramsX
            , paramsY = paramsY 
            , utX = utX
            , utY = utY );
            #, lambda = λ );                         # save the current model (cpu)
        global lowestMAE = meanMAE;                 # update previous lowest MAE
        @printf( "\tNew Model Saved with MAE: %s\n" 
               , formatted(meanMAE, :ENG, ndigits = 4) )
    end
    return [meanMSE meanMAE];                       # mean of error for all batches
end

### Run Training
numEpochs = 100;                                    # total number of epochs
lowestMAE = Inf;                                    # initialize MAE with ∞
errs = [];                                          # Array of training and validation losses

@epochs numEpochs push!(errs, trainModel())         # Run Training Loop for #epochs

# Reshape errors for a nice plot
losses = hcat( map((e) -> Float64(e[1]), errs)
             , map((e) -> Float64(e[2]), errs) );

# Plot Training Process
plot( 1:numEpochs, losses, lab = ["MSE" "MAE"]
    , xaxis=("# Epoch", (1,numEpochs))
    , yaxis=("Error", (0.0, ceil( max(losses...)
                                , digits = 3 )) ))

######################
## Evaluation
######################

## Use Current model ##
γ = γ |> cpu;

## Load specific model ##
modelPrefix = "./model/des-2021-02-08T17:41:45.706/ptmn90"
#modelPrefix = "./model/des-2021-02-08T18:11:21.781/ptmn90"
model = BSON.load(modelPrefix * ".bson");
γ = model[:model];
paramsX = model[:paramsX];
paramsY = model[:paramsY];
utX = model[:utX];
utY = model[:utY];
λ = model[:lambda];

dataDir = "../data/";
deviceName = "ptmn90";
mosFile = dataDir * deviceName * ".jld";
dataFrame = jldopen(mosFile, "r") do file
    file["database"];
end;
dataFrame.QVgs = dataFrame.Vgs.^2.0;
dataFrame.EVds = exp.(dataFrame.Vds);
dataFrame.Vgs = round.(dataFrame.Vgs, digits = 2);
dataFrame.gmid = dataFrame.gm ./ dataFrame.id;
dataFrame.A0 = dataFrame.gm ./ dataFrame.gds;
dataFrame.idW = dataFrame.id ./ dataFrame.W;
boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);
########################

### γ evaluation function for prediction characteristics
# predict(X) => y, where X = ["L", gmid", "vdsat", "Vds"];
# and y = [L Id gm/Id gm Vds ℯ.^Vds];
function predict(X)
    X′ = StatsBase.transform(utX, X)
    Y′ = γ(X′)
    Y = coxBox.(StatsBase.reconstruct(utY, Y′); λ = λ)
    return Float64.(Y)
end;

L = 300e-9;
Vds = 0.6;

traceT = sort( dataFrame[ ( (dataFrame.L .== L)
                         .& (dataFrame.W .== 2e-6)
                         .& (dataFrame.Vds .== Vds) )
                        , ["gmid", "idW"] ]
             , "idW" )

vdsat = 0.05:0.01:0.5;
len = length(vdsat);
Id = 25e-6;

x = [ fill(L, len)'
    ; fill(Id, len)'
    ; collect(vdsat)'
    ; fill(Vds, len)' 
    ; exp.(fill(Vds, len))' ]

y = predict(x);

traceP = sort( DataFrame(vdsat = vdsat, idW = y[1,:])
             , "vdsat" )

inspectdr();

plot( traceP.vdsat, traceP.idW; yscale = :log10
    , lab = "Approx", w = 2, xaxis = "vdsat", yaxis = "id/W" );
plot!(traceT.vdsat, traceT.idW, yscale = :log10, lab = "True", w = 2)

plot( plot(x[3,:], y[1,:], yscale = :log10)
    , plot(trace.vdsat, trace.idW, yscale = :log10)
    , layout = (1, 2))

