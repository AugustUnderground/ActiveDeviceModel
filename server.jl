#!/usr/bin/env -S julia --project
#
# $ curl -X POST -H "Content-Type: application/json" \
#        -d @./path/input.json http://host:port/predict
# $ curl -X POST -H "Content-Type: application/json" \
#        -d '{ "Vgs" : [ ... ]
#            , "Vds" : [ ...] 
#            , "Vbs" : [ 0 ... 0 ]
#            , "W"   : [ ...]
#            , "L"   : [ ... ] }'  \
#        http://host:port/predict

using ArgParse

function parseArgs()
    usg = """
You can POST a JSON file: 
`curl -X POST -H 'Content-Type: application/json' -d @./path/input.json http://host:port/predict` or put the data plain in the request: 
or 
`curl -X POST -H "Content-Type: application/json" \\
      -d '{ "Vgs" : [ ... ]
          , "Vds" : [ ...] 
          , "Vbs" : [ 0 ... 0 ]
          , "W"   : [ ...]
          , "L"   : [ ... ] }'  \\
      http://host:port/predict`"""

    settings = ArgParseSettings( usage = usg
                               , version = "0.0.1"
                               , add_version = true )

    @add_arg_table settings begin
        "--host", "-i"
            help = "IP Host Address on which the server listens."
            arg_type = String
            default = "0.0.0.0"
        "--port", "-p"
            help = "IP Host Address on which the server listens."
            arg_type = Int
            default = 8080
        "--quiet", "-q"
            help = "Be Quiet! Supress Verbose output."
            action = :store_true
        "model"
            help = "Model Direcotry. Should only contain 1 model (.bson) otherwise behaviour is undefined."
            arg_type = String
            required = true
    end

    return parse_args(settings)
end

args = parseArgs()

host = args["host"]
port = args["port"]
quiet = args["quiet"]
path = args["model"]

using HTTP
using JSON
using BSON
using PyCall
using ScikitLearn
using Flux
using Printf: @printf

joblib = pyimport("joblib");

struct Model
    net
    paramsX
    paramsY
    trafoX
    trafoY
end

function predict(model, req::HTTP.Request)
    io = IOBuffer(HTTP.payload(req))
    X = JSON.parse(io)
    println(X)

    Y = ((length(size(X)) < 2) ? [X'] : X') |>
         model.trafoX.transform |> 
         adjoint |> model.net |> adjoint |>
         model.trafoY.inverse_transform |> 
         adjoint

    Dict( ( model.paramsY[i] => Float(Y[i,:]) 
            for i = 1:length(model.paramsY) ) )

    HTTP.Response(200, JSON3.json(Y))
end

function loadModel(path::String)
    path = "./model/dev-2021-01-04T10:33:31.124/"
    files = readdir(path)

    modelFile = filter((f) -> endswith(f, "bson"), files) |> first
    model = BSON.load(path * modelFile)
    net = model[:model]
    paramsX = model[:paramsX]
    paramsY = model[:paramsY]

    trafoXFile = filter((f) -> endswith(f, "input"), files) |> first
    trafoYFile = filter((f) -> endswith(f, "output"), files) |> first
    trafoX = joblib.load(path * trafoXFile)
    trafoY = joblib.load(path * trafoYFile)

    return Model(net, paramsX, paramsY, trafoX, trafoY)
end

if !quiet
    @printf("Loading Model from %s\n", path)
end

model = loadModel(path)

router = HTTP.Router()
HTTP.@register( router, "POST", "predict"
              , (req) -> predict(model, req) )

if !quiet
    @printf("Starting Predict Server ...\nListening on %s:%s", host, port)
end

HTTP.serve(router, host, port)
