using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "2d_lattice", "MPS_TDVP_GRAPE.jl"))
run_grape_tdvp(2)
