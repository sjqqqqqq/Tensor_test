using Pkg
Pkg.activate("Tensor_test")

cd(joinpath(@__DIR__, "..", "2d_lattice"))
include("MPS_GRAPE.jl")
run_grape(3)
