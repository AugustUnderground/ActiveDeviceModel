# Use 🐍Call Packages (SkLearn) for Normalization
usePython = true;

#####################
## Dependencies
#####################

using Dates
using Plots
using StatsBase
using DataFrames
using JLD2
using BSON
using CUDA
using MLDataUtils
using Zygote
using Flux
using Flux: @epochs
using Logging
using Printf: @printf
using NumericIO

## 🐍 Deps
if usePython
    using PyCall
    if !PyCall.conda
        using Pkg
        using Conda
        ENV["PYTHON"] = ""
        Pkg.build("PyCall")
        Conda.add("scikit-learn")
    end
    using ScikitLearn
    joblib = pyimport("joblib");
end

######################
## Setup
######################

# File System
timeStamp = string(Dates.now());
dataDir = "../data/";
modelDir = "./model/dev-" * timeStamp * "/";
deviceName = "ptmn90";
mosFile = dataDir * deviceName * ".jld";
modelFile = modelDir * deviceName * ".bson";
trafoInFile = modelDir * deviceName * ".input";
trafoOutFile = modelDir * deviceName * ".output";
Base.Filesystem.mkdir(modelDir);

# Don't allow scalar operations on GPU
CUDA.allowscalar(false);

# Set Plot Backend
unicodeplots();
#inspectdr();

######################
## Data
######################

splitRatio = 0.8;
batchSize = 2500;

# Handle JLD2 dataframe
dataFrame = jldopen(mosFile, "r") do file
    file["database"];
end;

# Processing, Fitlering, Sampling and Shuffling
dataFrame.Vgs = round.(dataFrame.Vgs, digits = 2);

#sampleIdx = hcat( (0.5 .<= dataFrame.Vds .<= 0.7)
#                , (dataFrame.Vgs .∈ adjoint(0.0:0.1:1.2)));
#df = dataFrame[ (|).(sampleIdx...), :];

df = shuffleobs(dataFrame);

# Use all Parameters for training
paramsXY = names(dataFrame);
paramsX = filter((p) -> isuppercase(first(p)), paramsXY);
paramsY = filter((p) -> !in(p, paramsX), paramsXY);

# Number of In- and Outputs, for respective NN Layers
numX = length(paramsX);
numY = length(paramsY);

# Convert to Matrix for Flux
rawX = Matrix(df[:, paramsX ])';
rawY = Matrix(df[:, paramsY ])';

### Normalization / Scaling
if usePython
    @sk_import preprocessing: QuantileTransformer;
    trafoX = QuantileTransformer( output_distribution = "uniform"
                                , random_state = 666 );
    trafoY = QuantileTransformer( output_distribution = "uniform"
                                , random_state = 666 );
    dataX = rawX |> adjoint |> trafoX.fit_transform |> adjoint;
    dataY = rawY |> adjoint |> trafoY.fit_transform |> adjoint;
    joblib.dump(trafoX, trafoInFile)
    joblib.dump(trafoY, trafoOutFile)
else
    trafoX = fit( ZScoreTransform, rawX, dims = 2
                , center = true, scale = true );
    trafoY = fit( ZScoreTransform, rawY, dims = 2
                , center = true, scale = true );
    dataX = StatsBase.transform(trafoX, rawX);
    dataY = StatsBase.transform(trafoY, rawY);
end;

# Split data in training and validation set
trainX,validX = splitobs(dataX, splitRatio);
trainY,validY = splitobs(dataY, splitRatio);

# Create training and validation Batches
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
#θ₀ = deepcopy(θ);

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
            , weights = θ |> cpu );                 # save the current model (cpu)
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
modelFile = "./model/dev-2020-12-14T14:11:03.212/ptmn90.bson";
trafoInFile = "./model/dev-2020-12-14T14:11:03.212/ptmn90.input";
trafoOutFile = "./model/dev-2020-12-14T14:11:03.212/ptmn90.output";
model = BSON.load(modelFile);
φ = model[:model];
trafoX = joblib.load(trafoInFile);
trafoY = joblib.load(trafoOutFile);
######################

### φ evaluation function for prediction characteristics
if usePython
    predict(X) = begin
        rY = ((length(size(X)) < 2) ? [X'] : X') |>
             trafoX.transform |> 
             adjoint |> φ |> adjoint |>
             trafoY.inverse_transform |> 
             adjoint
        return Float64.(rY)
    end;
else
    predict(X) = begin
        tX = StatsBase.transform(trafoX, X);
        pY = tX |> gpu |> φ |> cpu;
        rY = StatsBase.reconstruct(trafoY, pY);
        return Float64.(rY)
    end;
end;

# Arbitrary Operating Point and sizing
W = 1.0e-6;
L = 3.0e-7;
vg = 0.6;
vd = 0.6;
vgs = 0.0:0.01:1.2;
vds = 0.0:0.01:1.2;

## Transfer Characterisitc

# Ground Truth from original Database
idtTrue = dataFrame[ ( (dataFrame.W .== W)
                    .& (dataFrame.L .== L)
                    .& (dataFrame.Vds .== vd))
                   , "id" ];

# Input matrix for φ according to paramsX
xt = [ collect(vgs)'
     ; repeat([vd], 121)'
     ; zeros(1, 121)
     ; repeat([W], 121)'
     ; repeat([L], 121)' ];

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
 
