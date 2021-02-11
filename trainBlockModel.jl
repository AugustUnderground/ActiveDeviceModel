#####################
## Dependencies
#####################

using Dates
using Random
using StatsBase
using Plots
using DataFrames
using JLD2
using BSON
using CUDA
using MLDataUtils
using Flux
using Flux: @epochs
using Printf: @printf
using NumericIO

######################
## Setup
######################

# File System
timeStamp = string(Dates.now());
dataDir = "../data/";
modelDir = "./model/blk-" * timeStamp * "/";
blockName = "cm-nxh035";
modelFile = modelDir * blockName * ".bson";
Base.Filesystem.mkdir(modelDir);

# Don't allow scalar operations on GPU
CUDA.allowscalar(false);

# Set Plot Backend
unicodeplots();
#inspectdr();

######################
## Data First Time
######################

# Load Sub-Sampled Dataset
blockFile = dataDir * blockName * "-subsample.jld";
dataFrame = jldopen((f) -> f["weighted"], blockFile, "r");

dataFrame.M     = dataFrame.Mcm12 ./ dataFrame.Mcm11;
dataFrame.Win   = dataFrame.Mcm11 .* dataFrame.W;
dataFrame.Wout  = dataFrame.Mcm12 .* dataFrame.W;

# Configure Traning Parameters
#paramsX = ["Ibias", "M", "Iout", "L", "M0_gm", "M1_gm"];               # ← this works
#paramsY1 = [ "Win", "Wout", "totalOutput_sigmaOut", "M0_vds", "O" ];   # ← this works

#paramsX = [ "Ibias", "M", "O", "M0_vdsat", "M0_fug" ]
#paramsY1 = [ "Win", "Wout", "L", "totalOutput_sigmaOut"
#           , "M0_vds", "M1_id", "M0_vth", "M0_gm", "M0_gds"
#           , "M1_vth", "M1_vdsat", "M1_gm", "M1_gds", "M1_fug" ]

paramsX = [ "Ibias", "L", "M", "O" , "M0_vdsat" ];
paramsY1 = [ "Win", "Wout", "M1_id", "M0_vds"
           , "M0_gm", "M0_gds", "M0_fug"
           , "M1_gm", "M1_gds", "M1_fug"
           , "totalOutput_sigmaOut" ];
paramsY2 = [ "M0_Avt_nmos_m__Contribution"
           , "M0_Avt_nmos_m__Sensitivity"
           , "M0_Au0_nmos_m__Contribution"
           , "M0_Au0_nmos_m__Sensitivity"
           , "M1_Avt_nmos_m__Contribution"
           , "M1_Avt_nmos_m__Sensitivity"
           , "M1_Au0_nmos_m__Contribution"
           , "M1_Au0_nmos_m__Sensitivity" ];
paramsY = [paramsY1 ; paramsY2];

# Number of In- and Outputs, for respective NN Layers
numX = length(paramsX);
numY = length(paramsY);

# Transform and Scale Data
boxCox(yᵢ; λ = 0.2) = (λ != 0) ? ((yᵢ.^λ) .- 1) ./ λ : log.(yᵢ) ;
coxBox(y′; λ = 0.2) = (λ != 0) ? exp.( log.((λ .* y′) .+ 1) ./ λ ) : exp.(y′) ;

λ = 0.2;

rawX = Matrix(dataFrame[:, paramsX ])';
rawY1 = Matrix(dataFrame[:, paramsY1])';
rawY2 = Matrix(dataFrame[:, paramsY2])';

bcX = boxCox.(rawX; λ = λ);
bcY = boxCox.(rawY1; λ = λ);

utX = StatsBase.fit(UnitRangeTransform, bcX; dims = 2, unit = true);
utY = StatsBase.fit(UnitRangeTransform, vcat(bcY, rawY2); dims = 2, unit = true);

dataX = StatsBase.transform(utX, bcX);
dataY = StatsBase.transform(utY, vcat(bcY, rawY2));

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
#φ = Chain( Dense(numX, 32, relu)
#         , Dense(32, 128, relu) 
#         , Dense(128, 512, relu) 
#         , Dense(512, 1024, relu) 
#         , Dense(1024, 512, relu) 
#         , Dense(512, 128, relu) 
#         , Dense(128, 64, relu) 
#         , Dense(64, numY, relu) ) |> gpu;

φ = Chain( Dense(numX, 128, relu)
         , Dense(128, 256, relu) 
         , Dense(256, 512, relu) 
         , Dense(512, 1024, relu) 
         , Dense(1024, 512, relu) 
         , Dense(512, 256, relu) 
         , Dense(256, 128, relu) 
         , Dense(128, numY, relu) ) |> gpu;

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
mse(x, y) = Flux.Losses.mse(φ(x), y, agg = mean);
mae(x, y) = Flux.Losses.mae(φ(x), y, agg = mean);

# Model Parameters (Weights)
θ = Flux.params(φ) |> gpu;

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
            , model = (φ |> cpu) 
            , paramsX = paramsX
            , paramsY1 = paramsY1
            , paramsY2 = paramsY2
            , utX = utX
            , utY = utY 
            , lambda = λ );                         # save the current model (cpu)
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

## Load specific model ##
#modelFile = "./model/2020-10-02T08:50:17.907/ptmn.bson";
model = BSON.load(modelFile);
φ = model[:model];
trafoX = model[:utX];
trafoY = model[:utY];
paramsX = model[:paramsX];
paramsY = model[:paramsY];
######################

φ = φ |> cpu;
######################

### φ evaluation function for prediction characteristics
function predict(X)
    Xbc = boxCox.(X; λ = λ);
    X′ = StatsBase.transform(utX, Xbc)
    Y′ = φ(X′)
    Yur = StatsBase.reconstruct(utY, Y′);
    Ycb = coxBox.(Yur[1:length(paramsY1),:]; λ = λ);
    Y = vcat(Ycb, Yur[(length(paramsY1) + 1):end,:]);
    return Float64.(Y)
end;

# Arbitrary Operating Point and sizing
sweepLen = 100;
vdsat = fill(0.2, sweepLen);
Ibias = fill(25e-6, sweepLen);
M = fill(2.0, sweepLen);
L = fill(7.5e-6, sweepLen);

#Ibias = range(5.0e-6, stop = 5.0e-5, length = sweepLen);
#vdsat = range(0.05, stop = 0.8, length = sweepLen);

x = [ Ibias M (M .* Ibias) L vdsat  vdsat ]';

y = predict(x)

Iout = y[first(indexin(["Iout"], paramsY)), :];

plot(Ibias, Iout)

# Prediction from φ
idtPred  = predict(xt)[first(indexin(["id"], paramsY)),:];

## Output Characterisitc
idoTrue = dataFrame[ ( (dataFrame.W .== W)
                    .& (dataFrame.L .== L)
                    .& (dataFrame.Vgs .== vg) )
                   , "id" ];

# Input matrix for φ according to paramsX
xo = [ repeat([vg], 121)'
     ; vds'
     ; zeros(1, 121)
     ; repeat([W], 121)'
     ; repeat([L], 121)' ];

# Prediction from φ 
idoPred = predict(xo)[first(indexin(["id"], paramsY)),:];

## Plot Results

# Plot Transfer Characterisitc
plot( vgs, [ idtTrue idtPred ]
    , xaxis=("V_gs", (0.0, 1.2))
    , yaxis=("I_d", (0.0, ceil( max(idtTrue...)
                              , digits = 4 )))
    , label=["tru" "prd"] )

# Plot Transfer Characterisitc
plot( vds, [ idoTrue idoPred ]
    , xaxis=("V_ds", (0.0, 1.2))
    , yaxis=("I_d", (0.0, ceil( max(idoTrue...)
                              , digits = 4 )))
    , label=["tru" "prd"])
 
