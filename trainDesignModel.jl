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
dataDir = "../data/";
modelDir = "./model/des-" * timeStamp * "/";
deviceName = "ptmn90";
mosFile = dataDir * deviceName * ".jld";
modelFile = modelDir * deviceName * ".bson";
trafoXFile = modelDir * deviceName * ".input";
trafoYFile = modelDir * deviceName * ".output";
Base.Filesystem.mkdir(modelDir);

# Don't allow scalar operations on GPU
CUDA.allowscalar(false);

# Set Plot Backend
unicodeplots();
#pyplot();

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
#dataFrame.fugLog = log10.(dataFrame.id ./ dataFrame.W);

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
paramsX = [ "L", "id", "gmid", "gm", "Vds" ]; #, "EVds" ]; # "vdsat" 
paramsY = [ "Vgs", "idW", "A0" ] #, "fug" ];
#paramsX = [ "W", "L", "Vgs", "QVgs", "Vds", "EVds"];
#paramsY = [ "idW", "gmid", "vdsat", "A0", "fug", "id", "gm", "gds"];

# Number of In- and Outputs, for respective NN Layers
numX = length(paramsX);
numY = length(paramsY);

# Convert to Matrix for Flux
rawX = Matrix(df[:, paramsX ])';
rawY = Matrix(df[:, paramsY ])';

boxCox(yᵢ; λ = 0.2) = λ != 0 ? (((yᵢ.^λ) .- 1) ./ λ) : log.(yᵢ);
coxBox(y′; λ = 0.2) = λ != 0 ? exp.(log.((λ .* y′) .+ 1) / λ) : exp.(y′);

λ = 0.2;
boxY = boxCox.(rawY; λ = λ);

utX = StatsBase.fit(UnitRangeTransform, rawX; dims = 2, unit = true); 
utY = StatsBase.fit(UnitRangeTransform, boxY; dims = 2, unit = true); 

dataX = StatsBase.transform(utX, rawX);
dataY = StatsBase.transform(utY, boxY);

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
#γ = Chain( Dense(numX,  64,     relu)
#         , Dense(64,    numY)
#         ) |> gpu;

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
            , utY = utY
            , lambda = λ );                  # save the current model (cpu)
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
modelPrefix = "./model/des-2021-01-12T13:58:59.001/ptmn90";
model = BSON.load(modelPrefix * ".bson");
γ = model[:model];
paramsX = model[:paramsX];
paramsY = model[:paramsY];
utX = model[:utX];
utY = model[:utY];
λ = model[:lambda];

### γ evaluation function for prediction characteristics
# predict(X) => y, where X = ["L", gmid", "vdsat", "Vds"];
# and y = [L Id gm/Id gm Vds ℯ.^Vds];
function predict(X)
    X′ = StatsBase.transform(utX, X)
    Y′ = γ(X′)
    Y = StatsBase.reconstruct(utY, coxBox.(Y′; λ = λ))
    return Float64.(Y)
end;

gmId = 1.0:0.35:25.0;
len = length(gmId);
Id = 50e-6;
Vds = 0.6;

x = [ fill(300e-9, 1, len)
    ; fill(Id, 1, len)
    ; collect(gmId)'
    ; (gmId .* Id)'
    ; fill(Vds, 1, len)
    ; exp.(fill(Vds, 1, len)) ];

y = predict(x);

idW = y[2,:];

pyplot();

plot(gmId, idW, yscale = :log10)

truPlt = plot();
for l in unique(df.L)
    gmid = dfR[ ( (dfR.L .== l)
               .& (dfR.Vds .== 0.6) )
              , "gmid" ];
    idW = dfR[ ( (dfR.L .== l)
               .& (dfR.Vds .== 0.6) )
              , "idW" ];
    truPlt = plot!(gmid, idW, yscale = :log10);
end
#truPlt

prdPlt = plot();
for l in unique(dfR.L)
    #gmid = 0.1:0.01:0.5
    gmid = 1:0.35:15.0
    len = first(size(gmid));
    x = [ repeat([l], len)'
        ; gmid'
        ; repeat([0.6], len)' ];
        #; exp.(repeat([0.6], len))' ];
    prd = predict(x)
    #idW = prd[5,:];
    idW = prd';
    prdPlt = plot!(gmid, idW, yaxis=:log10);
end
#prdPlt

plot(truPlt, prdPlt, layout = (1,2))
