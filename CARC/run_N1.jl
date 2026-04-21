using Pkg
Pkg.activate("Tensor_test")

include("MPS_GRAPE.jl")
run_grape(1)
