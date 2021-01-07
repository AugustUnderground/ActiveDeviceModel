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
using Flux
using Flux: @epochs
using Logging
using Printf: @printf
using NumericIO

# Use 🐍Call Packages (SkLearn) for Normalization
using PyCall
if !PyCall.conda
    using Pkg
    using Conda
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")
    Conda.add("scikit-learn")
end
using ScikitLearn
@sk_import preprocessing: QuantileTransformer;
joblib = pyimport("joblib");

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
#unicodeplots();
inspectdr();

######################
## Data
######################

# Handle JLD2 dataframe
dataFrame = jldopen(mosFile, "r") do file
    file["database"];
end;

# Processing, Fitlering, Sampling and Shuffling
dataFrame.Vgs = round.(dataFrame.Vgs, digits = 2);

dfW = dataFrame[ dataFrame.W .== 2e-6
              , ["gm", "id", "W", "L", "vdsat", "Vds", "Vgs", "gds", "fug"] ];

dfR = DataFrame( L = dfW.L
               , Vgs = dfW.Vgs
               , QVgs = (dfW.Vgs .^ 2.0)
               , Vds = dfW.Vds
               , EVds = exp.(dfW.Vds)
               , gmid = dfW.gm ./ dfW.id
               , vdsat = dfW.vdsat
               , A0 = dfW.gm ./ dfW.gds
               , fug = dfW.fug
               , idW = dfW.id ./ dfW.W 
               , A0Log = log.(dfW.gm ./ dfW.gds)
               , fugLog = log.(dfW.fug)
               , idWLog = log.(dfW.id ./ dfW.W) );

mask = (vec ∘ collect)(sum(Matrix(isinf.(dfR) .| isnan.(dfR)), dims = 2) .== 0);
df = shuffleobs(dfR[mask, :]);

# Use all Parameters for training
#paramsX = ["L", "gmid", "Vds", "EVds"];
#paramsY = ["A0", "A0Log", "fug", "fugLog", "idW", "idWLog", "Vgs", "QVgs"];
paramsX = ["L", "gmid", "Vds"];
paramsY = ["idW"];

# Number of In- and Outputs, for respective NN Layers
numX = length(paramsX);
numY = length(paramsY);

# Convert to Matrix for Flux
rawX = Matrix(df[:, paramsX ])';
rawY = Matrix(df[:, paramsY ])';

### Normalization / Scaling
trafoX = QuantileTransformer( output_distribution = "uniform"
                            , random_state = 666 );
trafoY = QuantileTransformer( output_distribution = "uniform"
                            , random_state = 666 );
dataX = rawX |> adjoint |> trafoX.fit_transform |> adjoint;
dataY = rawY |> adjoint |> trafoY.fit_transform |> adjoint;

joblib.dump(trafoX, trafoXFile);
joblib.dump(trafoY, trafoYFile);

# Split data in training and validation set
splitRatio = 0.7;
trainX,validX = splitobs(dataX, splitRatio);
trainY,validY = splitobs(dataY, splitRatio);

# Create training and validation Batches
batchSize = 500;
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
γ = Chain( Dense(numX,  32,     relu)
         , Dense(32,    8,    relu) 
         , Dense(8,    numY)
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
            , model = (γ |> cpu)
            , paramsX = paramsX
            , paramsY = paramsY );                  # save the current model (cpu)
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
γ = model[:model];
θ = model[:weights]
trafoX = joblib.load(trafoXFile)
trafoY = joblib.load(trafoYFile)
######################
γ = γ |> cpu;

### γ evaluation function for prediction characteristics
# predict(X) => y, where X = ["L", gmid", "vdsat", "Vds"];
# and y = ["A0", "idW", "fug"];
function predict(X)
    rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoX.transform |> 
         adjoint |> γ |> adjoint |>
         trafoY.inverse_transform |> 
         adjoint
    return Float64.(rY)
end

truPlt = plot();
for l in unique(dfR.L)
    gmid = dfR[ ( (dfR.L .== l)
               .& (dfR.Vds .== 0.6) )
              , "gmid" ];
    idW = dfR[ ( (dfR.L .== l)
               .& (dfR.Vds .== 0.6) )
              , "idW" ];
    truPlt = plot!(gmid, idW, yaxis=:log10);
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
