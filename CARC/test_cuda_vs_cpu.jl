# Smoke-test: confirm the CUDA port reproduces the CPU forward pass and
# gradient on a small problem before launching the real N=20 run.
#
# Usage on CARC:
#   julia --project=Tensor_test CARC/test_cuda_vs_cpu.jl
#
# Expects ≤ 1e-9 relative error on the forward state and ≤ 1e-8 max-abs
# error on the gradient.  Larger errors mean the port has a transcription bug.

using Pkg
Pkg.activate("Tensor_test")

include(joinpath(@__DIR__, "..", "2d_lattice", "exact_GRAPE_cuda.jl"))

using LinearAlgebra
using Printf
using Random
using CUDA

CUDA.functional() || error("CUDA not functional in this session")
@printf("CUDA device: %s\n", CUDA.name(CUDA.device()))

const N         = 2
const NUM_STEPS = 41
const T         = 2π
const DT        = T / (NUM_STEPS - 1)

Random.seed!(123)
ctrls = zeros(NUM_STEPS, 9)
for k in 1:6
    ctrls[:, k] .= 0.1 * randn(NUM_STEPS)
end
ctrls[:, 7] .= 1.0 .+ 0.1 * randn(NUM_STEPS)
ctrls[:, 8] .= 1.0 .+ 0.1 * randn(NUM_STEPS)
ctrls[:, 9] .= 1.0 .+ 0.1 * randn(NUM_STEPS)

# --- CPU reference ---------------------------------------------------------
sys_cpu = build_system(N)
psi0_cpu, psit_cpu = default_states(sys_cpu)

states_cpu   = trotter_fwd_store(sys_cpu, psi0_cpu, ctrls, DT)
overlap_cpu  = dot(psit_cpu, states_cpu[end])
costates_cpu = trotter_bwd_store(sys_cpu, psit_cpu, ctrls, DT)
grads_cpu    = compute_grads(sys_cpu, states_cpu, costates_cpu, ctrls, DT, overlap_cpu)
ψf_cpu       = states_cpu[end]

# --- CUDA path -------------------------------------------------------------
sys_cu = build_system_cuda(N)
psi0_cu, psit_cu = default_states_cuda(sys_cu)

states_cu   = trotter_fwd_store_cuda(sys_cu, psi0_cu, ctrls, DT)
overlap_cu  = dot(psit_cu, states_cu[end])
costates_cu = trotter_bwd_store_cuda(sys_cu, psit_cu, ctrls, DT)
grads_cu    = compute_grads_cuda(sys_cu, states_cu, costates_cu, ctrls, DT, overlap_cu)
ψf_cu       = Array(states_cu[end])

# --- Compare ---------------------------------------------------------------
fwd_err  = norm(ψf_cu - ψf_cpu) / norm(ψf_cpu)
ovl_err  = abs(overlap_cu - overlap_cpu)
grad_err = maximum(abs.(grads_cu .- grads_cpu))

@printf("\n— CPU vs CUDA comparison (N=%d, steps=%d) —\n", N, NUM_STEPS)
@printf("‖ψf_cu - ψf_cpu‖ / ‖ψf_cpu‖   = %.3e\n", fwd_err)
@printf("|overlap_cu - overlap_cpu|    = %.3e\n", ovl_err)
@printf("max |grads_cu - grads_cpu|    = %.3e\n", grad_err)

@assert fwd_err  < 1e-9 "Forward state mismatch"
@assert ovl_err  < 1e-9 "Overlap mismatch"
@assert grad_err < 1e-8 "Gradient mismatch"
println("All checks passed ✓")
