using Pkg
Pkg.activate("Tensor_test")

include(joinpath(@__DIR__, "..", "2d_lattice", "exact_GRAPE.jl"))

using JLD2
using Printf

const T_TOTAL  = 2π
const NUM_STEPS = 201
const MAX_ITER = 500

const DATA_DIR = joinpath(@__DIR__, "data")

function load_ctrls(N::Int, num_steps::Int)
    file = joinpath(DATA_DIR, "GRAPE_2d_Npair_$(N).jld2")
    isfile(file) || return nothing
    data = load(file)
    n = data["n"]
    @assert n == num_steps "stored n=$n disagrees with num_steps=$num_steps"
    ctrls = zeros(num_steps, 9)
    ctrls[:,1] = data["Va1"]; ctrls[:,2] = data["Va2"]; ctrls[:,3] = data["Va3"]
    ctrls[:,4] = data["Vb1"]; ctrls[:,5] = data["Vb2"]; ctrls[:,6] = data["Vb3"]
    ctrls[:,7] = data["U"];   ctrls[:,8] = data["Ja"];  ctrls[:,9] = data["Jb"]
    return ctrls
end

N = 13

println("\n" * "#"^70)
println("# Running N = $N")
println("#"^70)

if N == 1
    error("N=1 has no warm-start; use the N=1 random-init recipe instead.")
end

ctrls0 = load_ctrls(N - 1, NUM_STEPS)
isnothing(ctrls0) && error("warm-start file for N=$(N-1) not found in $DATA_DIR")
println("Warm-starting from N=$(N-1) pulse.")

sys = build_system(N)
@printf("Sector dims: Da = %d, Db = %d, D = %d\n", sys.Da, sys.Db, sys.D)
psi0, psi_target = default_states(sys)
dt = T_TOTAL / (NUM_STEPS - 1)
@printf("Time: T = %.4f, steps = %d, dt = %.4f\n\n", T_TOTAL, NUM_STEPS, dt)

ctrls_opt, final_fidelity = grape_2d_Npair(
    sys, psi0, psi_target, T_TOTAL, NUM_STEPS;
    ctrls0=ctrls0, max_iter=MAX_ITER, tol=1e-4, verbose=true
)
@printf("Final fidelity: %.8f\n", final_fidelity)

isdir(DATA_DIR) || mkpath(DATA_DIR)
output_file = joinpath(DATA_DIR, "GRAPE_2d_Npair_$(N).jld2")
println("Saving controls to $output_file ...")
n = NUM_STEPS
Va1, Va2, Va3 = ctrls_opt[:,1], ctrls_opt[:,2], ctrls_opt[:,3]
Vb1, Vb2, Vb3 = ctrls_opt[:,4], ctrls_opt[:,5], ctrls_opt[:,6]
U,   Ja,  Jb  = ctrls_opt[:,7], ctrls_opt[:,8], ctrls_opt[:,9]
fidelity = final_fidelity
Npair = N
T = T_TOTAL
@save output_file Npair n T dt Va1 Va2 Va3 Vb1 Vb2 Vb3 U Ja Jb fidelity
