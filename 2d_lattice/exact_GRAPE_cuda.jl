# exact_GRAPE_cuda.jl
# GPU port of exact_GRAPE.jl.  Reuses the CPU basis construction, indexing
# helpers, softplus reparam, and the run_npair pulse-init recipe; redefines
# only the hot-path arrays (evecs/evals/diag vectors/state vectors) on the GPU.
#
# Memory budget at N=20 (Da = C(23,3) = 1771, D = Da^2 ≈ 3.1M):
#   ψ on GPU                 :   50 MB  (ComplexF64)
#   evecs_a, evecs_b         :   50 MB  total
#   201 fwd + 201 bwd states : ≈ 20 GB  — fits comfortably on 40/80 GB GPU
#
# Memory budget at N=50 (Da = C(53,3) = 23426, D ≈ 5.5e8):
#   ψ on GPU                 :  8.8 GB
#   201 fwd + 201 bwd states : ≈ 3.5 TB — needs rematerialization
# That regime is out of scope of this file; the same factored kernels are
# correct, but the trotter_*_store functions would need a checkpoint policy.

using CUDA
using LinearAlgebra
using Printf
using Random
using Optim
using JLD2

include(joinpath(@__DIR__, "exact_GRAPE.jl"))

# ============================================================================
# GPU-backed system
# ============================================================================

struct System2DNpairCUDA
    N::Int
    Da::Int
    Db::Int
    D::Int
    basis_a::Vector{Vector{Int}}
    basis_b::Vector{Vector{Int}}
    evals_a::CuVector{Float64}
    evecs_a::CuMatrix{ComplexF64}
    evals_b::CuVector{Float64}
    evecs_b::CuMatrix{ComplexF64}
    dVa::Vector{CuVector{Float64}}
    dVb::Vector{CuVector{Float64}}
    dU::CuVector{Float64}
end

function build_system_cuda(N::Int)
    cpu = build_system(N)
    return System2DNpairCUDA(
        cpu.N, cpu.Da, cpu.Db, cpu.D,
        cpu.basis_a, cpu.basis_b,
        CuArray(cpu.evals_a), CuArray(cpu.evecs_a),
        CuArray(cpu.evals_b), CuArray(cpu.evecs_b),
        [CuArray(v) for v in cpu.dVa],
        [CuArray(v) for v in cpu.dVb],
        CuArray(cpu.dU),
    )
end

# Build a Fock-basis product state on the host, then upload.
function product_state_cuda(sys::System2DNpairCUDA, sites_a, sites_b)
    ψ = zeros(ComplexF64, sys.D)
    ia = species_index(sys.basis_a, sites_a)
    ib = species_index(sys.basis_b, sites_b)
    ψ[(ia - 1) * sys.Db + ib] = 1.0
    return CuVector{ComplexF64}(ψ)
end

function default_states_cuda(sys::System2DNpairCUDA)
    N = sys.N
    sA_L = fill(0, N); sB_L = fill(1, N)
    sA_R = fill(2, N); sB_R = fill(3, N)
    ψL = product_state_cuda(sys, sA_L, sB_L)
    ψR = product_state_cuda(sys, sA_R, sB_R)
    ψt = ψL .+ ψR
    ψt ./= norm(ψt)
    return copy(ψL), ψt
end

# ============================================================================
# Trotter propagation on GPU
# ============================================================================

function diag_exp_cuda(sys::System2DNpairCUDA, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt)
    h = Va1 .* sys.dVa[1] .+ Va2 .* sys.dVa[2] .+ Va3 .* sys.dVa[3] .+
        Vb1 .* sys.dVb[1] .+ Vb2 .* sys.dVb[2] .+ Vb3 .* sys.dVb[3] .+
        U   .* sys.dU
    return exp.(-im * dt .* h)
end

# Same reshape/matmul recipe as the CPU `apply_a` / `apply_b`; CUBLAS handles
# the GEMMs.  Returned vectors are fresh CuVectors (GC reclaims input ψ).
@inline function apply_a_cuda(psi::CuVector{ComplexF64}, Da, Db, pa, evecs_a)
    M = reshape(psi, Db, Da)
    return vec(((M * evecs_a) .* transpose(pa)) * evecs_a')
end

@inline function apply_b_cuda(psi::CuVector{ComplexF64}, Da, Db, pb, evecs_b)
    M = reshape(psi, Db, Da)
    return vec(evecs_b * (pb .* (evecs_b' * M)))
end

function trotter_step_cuda(sys::System2DNpairCUDA, psi,
                           Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt)
    Da, Db = sys.Da, sys.Db
    d1 = diag_exp_cuda(sys, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt / 2)
    pa = exp.(-im * Ja * dt .* sys.evals_a)
    pb = exp.(-im * Jb * dt .* sys.evals_b)
    psi = d1 .* psi
    psi = apply_a_cuda(psi, Da, Db, pa, sys.evecs_a)
    psi = apply_b_cuda(psi, Da, Db, pb, sys.evecs_b)
    psi = d1 .* psi
    return psi
end

function trotter_fwd_cuda(sys, psi0, ctrls, dt)
    psi = copy(psi0)
    for n in 1:size(ctrls, 1) - 1
        Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb = ctrls[n, :]
        psi = trotter_step_cuda(sys, psi, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt)
    end
    return psi
end

function trotter_fwd_store_cuda(sys, psi0, ctrls, dt)
    num_steps = size(ctrls, 1)
    states = Vector{CuVector{ComplexF64}}(undef, num_steps)
    psi = copy(psi0)
    states[1] = copy(psi)
    for n in 1:num_steps - 1
        Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb = ctrls[n, :]
        psi = trotter_step_cuda(sys, psi, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt)
        states[n + 1] = copy(psi)
    end
    return states
end

function trotter_bwd_store_cuda(sys, chi_T, ctrls, dt)
    num_steps = size(ctrls, 1)
    Da, Db = sys.Da, sys.Db
    costates = Vector{CuVector{ComplexF64}}(undef, num_steps)
    chi = copy(chi_T)
    costates[num_steps] = copy(chi)
    for n in num_steps - 1:-1:1
        Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb = ctrls[n, :]
        d1 = diag_exp_cuda(sys, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt / 2)
        pa = exp.(-im * Ja * dt .* sys.evals_a)
        pb = exp.(-im * Jb * dt .* sys.evals_b)
        chi = conj.(d1) .* chi
        chi = apply_b_cuda(chi, Da, Db, conj.(pb), sys.evecs_b)
        chi = apply_a_cuda(chi, Da, Db, conj.(pa), sys.evecs_a)
        chi = conj.(d1) .* chi
        costates[n] = copy(chi)
    end
    return costates
end

function compute_grads_cuda(sys, psi_states, chi_states, ctrls, dt, overlap)
    num_steps = size(ctrls, 1)
    grads = zeros(num_steps, 9)
    diag_vecs = (sys.dVa[1], sys.dVa[2], sys.dVa[3],
                 sys.dVb[1], sys.dVb[2], sys.dVb[3], sys.dU)
    Da, Db = sys.Da, sys.Db
    evecs_a = sys.evecs_a; evecs_b = sys.evecs_b
    evals_a = sys.evals_a; evals_b = sys.evals_b

    for n in 1:num_steps - 1
        Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb = ctrls[n, :]
        ψn   = psi_states[n]
        χnp1 = chi_states[n + 1]

        d1 = diag_exp_cuda(sys, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt / 2)
        pa = exp.(-im * Ja * dt .* evals_a)
        pb = exp.(-im * Jb * dt .* evals_b)

        ψ1 = d1 .* ψn
        ψ2 = apply_a_cuda(ψ1, Da, Db, pa, evecs_a)
        ψ3 = apply_b_cuda(ψ2, Da, Db, pb, evecs_b)

        dpa    = -im * dt .* evals_a .* pa
        dJa_ψ1 = apply_a_cuda(ψ1, Da, Db, dpa, evecs_a)
        gJa    = d1 .* apply_b_cuda(dJa_ψ1, Da, Db, pb, evecs_b)
        grads[n, 8] = 2 * real(conj(overlap) * dot(χnp1, gJa))

        dpb    = -im * dt .* evals_b .* pb
        dJb_ψ2 = apply_b_cuda(ψ2, Da, Db, dpb, evecs_b)
        gJb    = d1 .* dJb_ψ2
        grads[n, 9] = 2 * real(conj(overlap) * dot(χnp1, gJb))

        for (k, hk) in enumerate(diag_vecs)
            right  = (-im * dt / 2) .* hk .* (d1 .* ψ3)
            dD1_ψn = (-im * dt / 2) .* hk .* (d1 .* ψn)
            tmp    = apply_a_cuda(dD1_ψn, Da, Db, pa, evecs_a)
            tmp    = apply_b_cuda(tmp,    Da, Db, pb, evecs_b)
            left   = d1 .* tmp
            grads[n, k] = 2 * real(conj(overlap) * dot(χnp1, right .+ left))
        end
    end

    return grads
end

# ============================================================================
# GRAPE driver (GPU)
# ============================================================================

function grape_2d_Npair_cuda(sys::System2DNpairCUDA, psi0, psi_target, T, num_steps;
                             ctrls0=nothing, max_iter=500, tol=1e-4, verbose=true)

    dt = T / (num_steps - 1)

    ψ0 = isa(psi0, CuArray) ? CuVector{ComplexF64}(psi0) :
                              CuVector{ComplexF64}(Vector{ComplexF64}(psi0))
    ψt = isa(psi_target, CuArray) ? CuVector{ComplexF64}(psi_target) :
                                    CuVector{ComplexF64}(Vector{ComplexF64}(psi_target))

    if isnothing(ctrls0)
        ctrls = zeros(num_steps, 9)
        ctrls[:, 7] .= 1.0
        ctrls[:, 8] .= 1.0
        ctrls[:, 9] .= 1.0
    else
        ctrls = copy(ctrls0)
    end

    x_init = copy(ctrls)
    # Box constraint Ja, Jb ≥ 0 (cols 8, 9). Other channels unconstrained.
    x_init[:, 8] .= max.(x_init[:, 8], 0.0)
    x_init[:, 9] .= max.(x_init[:, 9], 0.0)
    lower = fill(-Inf, num_steps, 9); lower[:, 8] .= 0.0; lower[:, 9] .= 0.0
    upper = fill( Inf, num_steps, 9)

    @inline function unpack(x)
        u = reshape(x, num_steps, 9)
        c = copy(u)
        # c[:, 8] .= _softplus.(u[:, 8])
        # c[:, 9] .= _softplus.(u[:, 9])
        return u, c
    end

    function objective(x)
        _, c = unpack(x)
        ψf = trotter_fwd_cuda(sys, ψ0, c, dt)
        return 1.0 - abs2(dot(ψt, ψf))
    end

    function gradient!(g, x)
        u, c = unpack(x)
        states   = trotter_fwd_store_cuda(sys, ψ0, c, dt)
        overlap  = dot(ψt, states[end])
        costates = trotter_bwd_store_cuda(sys, ψt, c, dt)
        grads    = compute_grads_cuda(sys, states, costates, c, dt, overlap)
        # Chain rule for J = softplus(u) — disabled, J is now unconstrained.
        # grads[:, 8] .*= _sigmoid.(u[:, 8])
        # grads[:, 9] .*= _sigmoid.(u[:, 9])
        g .= -vec(grads)
        # Release the large state/costate arrays before the next L-BFGS step
        states = nothing
        costates = nothing
        GC.gc(false)
    end

    x0 = vec(x_init)
    iter_count = Ref(0)

    function cb(state)
        iter_count[] += 1
        fid = 1.0 - state.f_x
        if verbose && (iter_count[] == 1 || iter_count[] % 10 == 0)
            @printf("Iter %4d: fidelity = %.8f\n", iter_count[], fid)
        end
        if fid > 0.99
            verbose && @printf("Iter %4d: fidelity = %.8f ≥ 0.99, stopping.\n",
                               iter_count[], fid)
            return true
        end
        return false
    end

    result = optimize(
        objective, gradient!,
        vec(lower), vec(upper), x0,
        Fminbox(LBFGS(m=20)),
        Optim.Options(
            iterations = max_iter,
            g_tol      = tol * 1e-2,
            f_reltol   = tol * 1e-2,
            show_trace = false,
            callback   = cb
        )
    )

    u_opt = reshape(Optim.minimizer(result), num_steps, 9)
    ctrls_opt = copy(u_opt)
    # ctrls_opt[:, 8] .= _softplus.(u_opt[:, 8])
    # ctrls_opt[:, 9] .= _softplus.(u_opt[:, 9])
    final_fidelity = 1.0 - Optim.minimum(result)

    if verbose
        println("\nL-BFGS completed:")
        println("  Iterations: $(Optim.iterations(result))")
        println("  Converged:  $(Optim.converged(result))")
    end

    return ctrls_opt, final_fidelity
end

# ============================================================================
# Convenience driver mirroring run_npair
# ============================================================================

function run_npair_cuda(N::Int = 1;
                        T = 2π, num_steps = 201, seed = 42, max_iter = 500,
                        save = true, psi0 = nothing, psi_target = nothing,
                        verbose = true, ctrls0 = nothing)
    println("="^60)
    println("GRAPE (CUDA, Trotter) — 2D Lattice, N = $N pair(s)")
    println("="^60)

    if !CUDA.functional()
        error("CUDA is not functional in this Julia session.")
    end
    @printf("CUDA device: %s\n", CUDA.name(CUDA.device()))

    sys = build_system_cuda(N)
    @printf("Sector dims: Da = %d, Db = %d, D = %d\n", sys.Da, sys.Db, sys.D)

    if isnothing(psi0) || isnothing(psi_target)
        psi0_def, ψt_def = default_states_cuda(sys)
        psi0       = something(psi0, psi0_def)
        psi_target = something(psi_target, ψt_def)
    end

    dt = T / (num_steps - 1)
    @printf("Time: T = %.4f,  steps = %d,  dt = %.4f\n\n", T, num_steps, dt)

    if isnothing(ctrls0)
        Random.seed!(seed)
        t_arr = collect(range(0, T, length = num_steps))
        ctrls0 = zeros(num_steps, 9)
        for k in 1:6
            ctrls0[:, k] .= 0.1 * randn(num_steps)
        end
        ctrls0[:, 7] .= 1.0 .+ 0.1 * randn(num_steps)
        ctrls0[:, 8] .= 1.0 .+ 0.2 * sin.(2π .* t_arr ./ T) .+ 0.1 * randn(num_steps)
        ctrls0[:, 9] .= 1.0 .+ 0.2 * cos.(2π .* t_arr ./ T) .+ 0.1 * randn(num_steps)
    else
        @assert size(ctrls0) == (num_steps, 9) "ctrls0 must be ($(num_steps), 9), got $(size(ctrls0))"
        ctrls0 = copy(ctrls0)
        println("Using provided ctrls0 (warm start).")
    end

    println("Starting GRAPE optimization (GPU)...")
    println("-"^60)
    ctrls_opt, final_fidelity = grape_2d_Npair_cuda(
        sys, psi0, psi_target, T, num_steps;
        ctrls0 = ctrls0, max_iter = max_iter, tol = 1e-4, verbose = verbose,
    )
    println("-"^60)
    @printf("Final fidelity: %.8f\n", final_fidelity)

    ψf = trotter_fwd_cuda(sys, psi0, ctrls_opt, dt)
    @printf("|⟨ψ_target|ψ_final⟩|² = %.8f\n", abs2(dot(psi_target, ψf)))

    if save
        output_file = "data/GRAPE_2d_Npair_$(N).jld2"
        println("\nSaving controls to $output_file ...")
        n = num_steps
        Va1, Va2, Va3 = ctrls_opt[:, 1], ctrls_opt[:, 2], ctrls_opt[:, 3]
        Vb1, Vb2, Vb3 = ctrls_opt[:, 4], ctrls_opt[:, 5], ctrls_opt[:, 6]
        U,   Ja,  Jb  = ctrls_opt[:, 7], ctrls_opt[:, 8], ctrls_opt[:, 9]
        fidelity = final_fidelity
        Npair = N
        @save output_file Npair n T dt Va1 Va2 Va3 Vb1 Vb2 Vb3 U Ja Jb fidelity
    end

    return ctrls_opt, final_fidelity, sys
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 1
    run_npair_cuda(N)
end
