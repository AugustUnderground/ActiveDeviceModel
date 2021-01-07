### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 65c8e756-3ab4-11eb-1807-1381a5ba43a7
using PlutoUI, DifferentialEquations, DataDrivenDiffEq, Plots, ModelingToolkit, LinearAlgebra, Calculus, Flux, NNlib, CUDA, Zygote, PyCall, ScikitLearn, BSON, Optim

# ╔═╡ af845ec6-4054-11eb-2c4e-db69e349a432
begin
	Core.eval(Main, :(using PyCall))
	Core.eval(Main, :(using Zygote))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
end

# ╔═╡ c8caf49e-405b-11eb-2432-650948c4e9d6
md"""
# Find OP of Circuit
"""

# ╔═╡ c152548e-4054-11eb-2e87-17f486198fdd
joblib = pyimport("joblib");

# ╔═╡ ca6260c8-4054-11eb-1e72-3dfc7c259819
begin
	modelPath = "./model/dev-2020-12-16T16:14:05.641/ptmn90";
	modelFile = modelPath * ".bson";
	trafoInFile = modelPath * ".input";
	trafoOutFile = modelPath * ".output";
	model = BSON.load(modelFile);|
	φ = model[:model];
	trafoX = joblib.load(trafoInFile);
	trafoY = joblib.load(trafoOutFile);
	paramsX = ["Vgs", "Vds", "Vbs", "W", "L", "eVgs", "eVds" ];
	paramsY = [ "vth", "vdsat", "id", "gm", "gmb","gds", "fug"
			  , "cgd", "cgb", "cgs", "cds", "csb", "cdb"
			  , "idW", "gmid", "a0" ];
end;

# ╔═╡ d06e4784-4054-11eb-0f9c-93fca889c367
function predict(X)
 	rY = ((length(size(X)) < 2) ? [X'] : X') |>
         trafoX.transform |> 
         adjoint |> φ |> adjoint |>
         trafoY.inverse_transform |> 
         adjoint
  	return Float64.(rY)
end;

# ╔═╡ d4ec9694-4054-11eb-285f-f3cae3463171
nmos = (vgs, vds, vbs, w, l) -> predict([vgs, vds, vbs, w, l, vgs^2, vds^(1/2)]);

# ╔═╡ e079208e-405b-11eb-37a4-59d248aac331
begin
	VDD = 1.2;
	VSS = 0.0;
	Vbias = 0.5;
	Vin = 0.6;
	Vx = 0.0;

md"""
The _initial condition_ $\varphi_{init}$ is defined as follows:

$$\varphi_{init} = 
\begin{bmatrix} 
	\varphi_{DD} = 1.2 V\\
	\varphi_{SS} = 0.0 V\\
	\varphi_{bias} = 0.8 V\\
	\varphi_{in} = 0.5 V\\
	\varphi_{x} = 0.6 V\\
\end{bmatrix}$$
"""
end

# ╔═╡ c9ef41ee-4057-11eb-0d00-c35296502b16
function CascodeOP(φₓ)
	[ nmos(Vbias, (VDD - φₓ), 0.0, 3e-7, 1.5e-6)
 	, nmos(Vin, φₓ, 0.0, 3e-7, 1.5e-6) ]
end;

# ╔═╡ 8cb1bb0e-4057-11eb-04be-81ca0370d22a
function kclError(Vₓ)
	i₁, i₂ = [ op[first(indexin(["id"], paramsY))] 
			   for op in CascodeOP(first(Vₓ)) ].^(-1);
	return abs(i₁ - i₂)
end;

# ╔═╡ 0d641b94-4059-11eb-325d-ab66cd811454
opt = Optim.optimize(kclError, [VSS], [VDD], [Vx], Fminbox(GradientDescent()); autodiff = :finite)

# ╔═╡ 06546262-405e-11eb-266f-2b383405351f
φ̂ₓ = Optim.minimizer(opt)

# ╔═╡ Cell order:
# ╠═c8caf49e-405b-11eb-2432-650948c4e9d6
# ╠═65c8e756-3ab4-11eb-1807-1381a5ba43a7
# ╠═af845ec6-4054-11eb-2c4e-db69e349a432
# ╠═c152548e-4054-11eb-2e87-17f486198fdd
# ╠═ca6260c8-4054-11eb-1e72-3dfc7c259819
# ╠═d06e4784-4054-11eb-0f9c-93fca889c367
# ╠═d4ec9694-4054-11eb-285f-f3cae3463171
# ╠═e079208e-405b-11eb-37a4-59d248aac331
# ╠═c9ef41ee-4057-11eb-0d00-c35296502b16
# ╠═8cb1bb0e-4057-11eb-04be-81ca0370d22a
# ╠═0d641b94-4059-11eb-325d-ab66cd811454
# ╠═06546262-405e-11eb-266f-2b383405351f
