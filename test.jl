using Predict

##### CONDA STUFF
using Conda
Conda.add("scikit-learn")

## USING IT
using ScikitLearn

@sk_import preprocessing: QuantileTransformer

## Data
using DataFrames
using CSV
using JLD2
using Plots

jldFile = "../data/ptmn45.jld"
dataFrame = jldopen(jldFile, "r") do file
    file["database"];
end;


csvFile = "../data/ptmn.csv";
csvData = CSV.File( csvFile
                  , header=true
                  , type=Float64 
                  ) |> DataFrame;
csvData.vgs = round.(csvData.vgs, digits = 2);

df =  csvData[ ( (csvData.vds .== 0.6)
              .& (csvData.L .== 5.5e-7 )
              .& (csvData.W .== 8.0e-7 ) )
             , ["W", "L", "gm", "gds", "id", "vdsat"]];

df.idw = df.id ./ df.W;
df.gmid = df.gm ./ df.id;

df[:,["gm", "id", "gmid"]]

jldFile = "../data/ptmn.jld";
jldData = jldopen(jldFile, "r") do file
    file["database"];
end;
jldData.Vgs = round.(jldData.Vgs, digits = 2);

pd =  jldData[ ( (jldData.Vds .== 0.6)
              .& (jldData.L .== 5.5e-7 )
              .& (jldData.W .== 8.0e-7 ) )
             , ["W", "L", "gm", "gds", "id", "vdsat"]];

pd.idw = pd.id ./ pd.W;
pd.gmid = pd.gm ./ pd.id;

pd[:,["gm", "id", "gmid"]]

plot(pd.gmid, pd.idw, yscale = :log10)
