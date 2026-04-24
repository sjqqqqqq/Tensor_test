# 2d_lattice_GRAPE_Npair.jl
# GRAPE optimal control for 2D two-species lattice with N pairs (SOFT-CORE BOSONS).
# Generalization of 2d_lattice_GRAPE.jl: N identical a-type bosons and
# N identical b-type bosons on the same 4-site ring, with no on-site occupation cap
# (each site may host arbitrarily many particles per species).
#
# Hilbert space:
#   Per species: all occupation tuples (n_0, n_1, n_2, n_3) with Σn_j = N,
#                 n_j ≥ 0.  Dimension per species is C(N+3, 3).
#   Da = Db = C(N+3, 3),  D = Da * Db,  flat index (ai-1)*Db + bi.
#   Basis is enumerated so that for N=1 it coincides with the hard-core /
#   single-particle basis of 2d_lattice_GRAPE.jl.
#
# Hamiltonian (same 9 controls as original):
#   H(t) = Va1·H_Va1 + Va2·H_Va2 + Va3·H_Va3
#        + Vb1·H_Vb1 + Vb2·H_Vb2 + Vb3·H_Vb3
#        + U·H_U + Ja·H_Ja + Jb·H_Jb
#
# Where
#   H_Vak = N^a_{k} - N^a_{0}       (site-resolved number operators, diagonal)
#   H_U   = Σ_j N^a_j · N^b_j       (same-site a–b interaction, diagonal)
#   H_Ja  = hopping on species a (sum over bonds (i,j): c†_{a,i} c_{a,j} + h.c.)
#   H_Jb  = hopping on species b.
#
# Bonds of the 4-site square ring (matching Gamma4 in the original):
#   (0,1), (0,2), (1,3), (2,3).
#
# Hopping (bosonic, no Jordan-Wigner string):
#   c†_i c_j |..., n_i, n_j, ...⟩ = √((n_i+1)·n_j) |..., n_i+1, n_j-1, ...⟩
#
# Trotter split (identical to the original):
#   H₁ = diagonal part; H_Ja and H_Jb commute since they act on different species.
#   exp(-iHdt) ≈ exp(-iH₁dt/2) · exp(-iJa·H_Ja·dt) · exp(-iJb·H_Jb·dt) · exp(-iH₁dt/2)
#
# For N=1 the √-factors all reduce to 1 and the basis has 4 states
# [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] in that order — matrices are identical
# to 2d_lattice_GRAPE.jl, so the optimizer trajectory is bit-for-bit the same.

using LinearAlgebra
using Printf
using Random
using Optim
using JLD2

# ============================================================================
# System definition (parameterised by N)
# ============================================================================

const L_SITES = 4
const BONDS   = [(0,1), (0,2), (1,3), (2,3)]   # Gamma4 nonzero entries

"""
All occupation tuples (n_0, …, n_{L-1}) with Σ n_j = N (soft-core bosons).
Enumeration order is anti-lexicographic on the leading coordinate, so the
first-emitted states are those most weighted on site 0, then site 1, etc.
For N=1 this returns [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]].
"""
function species_basis(L::Int, N::Int)
    states = Vector{Vector{Int}}()
    buf = zeros(Int, L)
    function rec(remaining::Int, idx::Int)
        if idx == L
            buf[idx] = remaining
            push!(states, copy(buf))
            return
        end
        for k in remaining:-1:0            # anti-lex: high first
            buf[idx] = k
            rec(remaining - k, idx + 1)
        end
    end
    rec(N, 1)
    return states
end

"""Bosonic hopping matrix on BONDS: amplitude √((n_i+1)·n_j) for c†_i c_j."""
function species_hopping(basis::Vector{Vector{Int}})
    D = length(basis)
    H = zeros(ComplexF64, D, D)
    idx = Dict{Vector{Int},Int}(basis[k] => k for k in 1:D)
    for k in 1:D
        state = basis[k]
        for (i, j) in BONDS
            ii = i + 1; jj = j + 1
            # c†_i c_j : requires n_j ≥ 1
            if state[jj] > 0
                new_state = copy(state)
                new_state[jj] -= 1
                new_state[ii] += 1
                H[idx[new_state], k] += sqrt((state[ii] + 1) * state[jj])
            end
            # c†_j c_i : requires n_i ≥ 1
            if state[ii] > 0
                new_state = copy(state)
                new_state[ii] -= 1
                new_state[jj] += 1
                H[idx[new_state], k] += sqrt((state[jj] + 1) * state[ii])
            end
        end
    end
    return H
end

"""Diagonal vector of n_site in the soft-core basis (integers cast to Float64)."""
function species_occupation(basis::Vector{Vector{Int}}, site::Int)
    return [Float64(s[site + 1]) for s in basis]
end

struct System2DNpair
    N::Int
    Da::Int
    Db::Int
    D::Int
    basis_a::Vector{Vector{Int}}
    basis_b::Vector{Vector{Int}}
    evals_a::Vector{Float64}
    evecs_a::Matrix{ComplexF64}
    evals_b::Vector{Float64}
    evecs_b::Matrix{ComplexF64}
    dVa::Vector{Vector{Float64}}  # length 3: Va1, Va2, Va3
    dVb::Vector{Vector{Float64}}  # length 3: Vb1, Vb2, Vb3
    dU::Vector{Float64}
end

function build_system(N::Int)
    @assert N ≥ 1 "N must be ≥ 1"
    basis_a = species_basis(L_SITES, N)
    basis_b = species_basis(L_SITES, N)
    Da = length(basis_a); Db = length(basis_b); D = Da*Db

    # Eigendecompose only the small Da×Da / Db×Db hopping factors, never the
    # D×D Kronecker. At N=10, D² complex would be ≈ 107 GB; h_a is 286×286.
    h_a = species_hopping(basis_a)
    h_b = species_hopping(basis_b)
    Fa = eigen(Hermitian(Matrix(h_a)))
    Fb = eigen(Hermitian(Matrix(h_b)))
    evals_a = Fa.values
    evecs_a = Matrix{ComplexF64}(Fa.vectors)
    evals_b = Fb.values
    evecs_b = Matrix{ComplexF64}(Fb.vectors)

    n_a = [species_occupation(basis_a, j) for j in 0:L_SITES-1]
    n_b = [species_occupation(basis_b, j) for j in 0:L_SITES-1]

    # Relative-potential diagonals: V_k = n_{site k} - n_{site 0}
    dVa = [kron(n_a[k+1] .- n_a[1], ones(Db)) for k in 1:3]
    dVb = [kron(ones(Da), n_b[k+1] .- n_b[1]) for k in 1:3]

    # Interaction diagonal: Σ_j n^a_j · n^b_j
    dU = zeros(Float64, D)
    for j in 1:L_SITES
        dU .+= kron(n_a[j], n_b[j])
    end

    return System2DNpair(N, Da, Db, D, basis_a, basis_b,
                         evals_a, evecs_a, evals_b, evecs_b,
                         dVa, dVb, dU)
end

# Index helpers -------------------------------------------------------------

"""
Return the basis index of the occupation state obtained by placing particles at
`sites` (a list of 0-indexed site labels; repeats allowed for multi-occupation).
"""
function species_index(basis::Vector{Vector{Int}}, sites)
    n = zeros(Int, L_SITES)
    for s in sites
        @assert 0 ≤ s < L_SITES "site $s out of 0..$(L_SITES-1)"
        n[s + 1] += 1
    end
    k = findfirst(v -> v == n, basis)
    isnothing(k) && error("occupation $n not in basis (check particle number)")
    return k
end

flat_index(sys::System2DNpair, sites_a, sites_b) =
    (species_index(sys.basis_a, sites_a) - 1) * sys.Db +
     species_index(sys.basis_b, sites_b)

function product_state(sys::System2DNpair, sites_a, sites_b)
    ψ = zeros(ComplexF64, sys.D)
    ψ[flat_index(sys, sites_a, sites_b)] = 1.0
    return ψ
end

# ============================================================================
# Trotter propagation (mirrors 2d_lattice_GRAPE.jl)
# ============================================================================

function diag_exp(sys::System2DNpair, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt)
    h = Va1 .* sys.dVa[1] .+ Va2 .* sys.dVa[2] .+ Va3 .* sys.dVa[3] .+
        Vb1 .* sys.dVb[1] .+ Vb2 .* sys.dVb[2] .+ Vb3 .* sys.dVb[3] .+
        U   .* sys.dU
    return exp.(-im * dt .* h)
end

# Factored hopping-propagator application. psi is reshaped as (Db, Da)
# (column-major, flat index (ai-1)*Db + bi).
#
#   kron(h_a, I_b)·ψ   ⇔   M → M · h_a         (right-multiplication)
#   kron(I_a, h_b)·ψ   ⇔   M → h_b · M         (left-multiplication)
#
# With h_a = V_a · diag(λ_a) · V_a†, the exponential exp(-iJa·dt·h_a) has
# the eigendecomp V_a · diag(pa) · V_a† with pa = exp(-iJa·dt·λ_a); swap pa
# for any scalar function of λ_a to apply other analytic functions of h_a.
@inline function apply_a(psi, Da, Db, pa, evecs_a)
    M = reshape(psi, Db, Da)
    return vec(((M * evecs_a) .* transpose(pa)) * evecs_a')
end

@inline function apply_b(psi, Da, Db, pb, evecs_b)
    M = reshape(psi, Db, Da)
    return vec(evecs_b * (pb .* (evecs_b' * M)))
end

function trotter_step2d(sys, psi, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt)
    Da, Db = sys.Da, sys.Db
    d1 = diag_exp(sys, Va1, Va2, Va3, Vb1, Vb2, Vb3, U, dt/2)
    pa = exp.(-im*Ja*dt .* sys.evals_a)
    pb = exp.(-im*Jb*dt .* sys.evals_b)
    psi = d1 .* psi
    psi = apply_a(psi, Da, Db, pa, sys.evecs_a)
    psi = apply_b(psi, Da, Db, pb, sys.evecs_b)
    psi = d1 .* psi
    return psi
end

function trotter_fwd(sys, psi0, ctrls, dt)
    psi = copy(psi0)
    for n in 1:size(ctrls,1)-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = trotter_step2d(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
    end
    return psi
end

function trotter_fwd_store(sys, psi0, ctrls, dt)
    num_steps = size(ctrls,1)
    states = Vector{Vector{ComplexF64}}(undef, num_steps)
    psi = copy(psi0)
    states[1] = copy(psi)
    for n in 1:num_steps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        psi = trotter_step2d(sys, psi, Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb, dt)
        states[n+1] = copy(psi)
    end
    return states
end

function trotter_bwd_store(sys, chi_T, ctrls, dt)
    num_steps = size(ctrls,1)
    Da, Db = sys.Da, sys.Db
    costates = Vector{Vector{ComplexF64}}(undef, num_steps)
    chi = copy(chi_T)
    costates[num_steps] = copy(chi)
    for n in num_steps-1:-1:1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        d1 = diag_exp(sys, Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
        pa = exp.(-im*Ja*dt .* sys.evals_a)
        pb = exp.(-im*Jb*dt .* sys.evals_b)
        chi = conj.(d1) .* chi
        chi = apply_b(chi, Da, Db, conj.(pb), sys.evecs_b)
        chi = apply_a(chi, Da, Db, conj.(pa), sys.evecs_a)
        chi = conj.(d1) .* chi
        costates[n] = copy(chi)
    end
    return costates
end

function compute_grads(sys, psi_states, chi_states, ctrls, dt, overlap)
    num_steps = size(ctrls,1)
    grads = zeros(num_steps, 9)
    diag_vecs = (sys.dVa[1], sys.dVa[2], sys.dVa[3],
                 sys.dVb[1], sys.dVb[2], sys.dVb[3], sys.dU)
    Da, Db = sys.Da, sys.Db
    evecs_a = sys.evecs_a; evecs_b = sys.evecs_b
    evals_a = sys.evals_a; evals_b = sys.evals_b

    for n in 1:num_steps-1
        Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
        ψn   = psi_states[n]
        χnp1 = chi_states[n+1]

        d1 = diag_exp(sys, Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
        pa = exp.(-im*Ja*dt .* evals_a)
        pb = exp.(-im*Jb*dt .* evals_b)

        ψ1 = d1 .* ψn
        ψ2 = apply_a(ψ1, Da, Db, pa, evecs_a)
        ψ3 = apply_b(ψ2, Da, Db, pb, evecs_b)

        # Ja gradient: ∂_Ja exp(-iJa·dt·h_a) = -i·dt·h_a·exp(-iJa·dt·h_a)
        dpa    = -im*dt .* evals_a .* pa
        dJa_ψ1 = apply_a(ψ1, Da, Db, dpa, evecs_a)
        gJa    = d1 .* apply_b(dJa_ψ1, Da, Db, pb, evecs_b)
        grads[n, 8] = 2 * real(conj(overlap) * dot(χnp1, gJa))

        # Jb gradient
        dpb    = -im*dt .* evals_b .* pb
        dJb_ψ2 = apply_b(ψ2, Da, Db, dpb, evecs_b)
        gJb    = d1 .* dJb_ψ2
        grads[n, 9] = 2 * real(conj(overlap) * dot(χnp1, gJb))

        # Diagonal-control gradients (Va1,Va2,Va3,Vb1,Vb2,Vb3,U)
        for (k, hk) in enumerate(diag_vecs)
            right  = (-im*dt/2) .* hk .* (d1 .* ψ3)
            dD1_ψn = (-im*dt/2) .* hk .* (d1 .* ψn)
            tmp    = apply_a(dD1_ψn, Da, Db, pa, evecs_a)
            tmp    = apply_b(tmp,    Da, Db, pb, evecs_b)
            left   = d1 .* tmp
            grads[n, k] = 2 * real(conj(overlap) * dot(χnp1, right .+ left))
        end
    end

    return grads
end

# ============================================================================
# GRAPE optimization
# ============================================================================

function grape_2d_Npair(sys::System2DNpair, psi0, psi_target, T, num_steps;
                        ctrls0=nothing, max_iter=500, tol=1e-4, verbose=true)

    dt = T / (num_steps - 1)

    ψ0 = Vector{ComplexF64}(psi0)
    ψt = Vector{ComplexF64}(psi_target)

    if isnothing(ctrls0)
        ctrls = zeros(num_steps, 9)
        ctrls[:, 7] .= 1.0
        ctrls[:, 8] .= 1.0
        ctrls[:, 9] .= 1.0
    else
        ctrls = copy(ctrls0)
    end

    function objective(x)
        c = reshape(x, num_steps, 9)
        ψf = trotter_fwd(sys, ψ0, c, dt)
        return 1.0 - abs2(dot(ψt, ψf))
    end

    function gradient!(g, x)
        c = reshape(x, num_steps, 9)
        states   = trotter_fwd_store(sys, ψ0, c, dt)
        overlap  = dot(ψt, states[end])
        costates = trotter_bwd_store(sys, ψt, c, dt)
        grads    = compute_grads(sys, states, costates, c, dt, overlap)
        g .= -vec(grads)
    end

    x0 = vec(ctrls)
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
        objective, gradient!, x0,
        LBFGS(m=20),
        Optim.Options(
            iterations = max_iter,
            g_tol      = tol * 1e-2,
            f_reltol   = tol * 1e-2,
            show_trace = false,
            callback   = cb
        )
    )

    ctrls_opt      = reshape(Optim.minimizer(result), num_steps, 9)
    final_fidelity = 1.0 - Optim.minimum(result)

    if verbose
        println("\nL-BFGS completed:")
        println("  Iterations: $(Optim.iterations(result))")
        println("  Converged:  $(Optim.converged(result))")
    end

    return ctrls_opt, final_fidelity
end

# ============================================================================
# Default SPDC-like states for N pairs
# ============================================================================
#
# For N=1 we reproduce 2d_lattice_GRAPE.jl exactly:
#   initial = |a@{0},        b@{1}⟩
#   target  = (|a@{0},b@{1}⟩ + |a@{2},b@{3}⟩)/√2
#
# For N>1 we pair up "left bond" (0,1) and "right bond" (2,3) symmetrically.
# Feel free to override by passing custom psi0 / psi_target to `run_npair`.

"""
Default soft-core SPDC-like states for N pairs.
  Initial: all N a-bosons on site 0, all N b-bosons on site 1       (bond (0,1))
  Target : (|all-a@0, all-b@1⟩ + |all-a@2, all-b@3⟩)/√2             (bond (0,1) ⊕ (2,3))
For N=1 this reduces to |a@0, b@1⟩ → (|a@0,b@1⟩ + |a@2,b@3⟩)/√2, matching
2d_lattice_GRAPE.jl exactly.
"""
function default_states(sys::System2DNpair)
    N = sys.N
    sA_L = fill(0, N); sB_L = fill(1, N)
    sA_R = fill(2, N); sB_R = fill(3, N)
    ψL = product_state(sys, sA_L, sB_L)
    ψR = product_state(sys, sA_R, sB_R)
    ψt = ψL .+ ψR
    ψt ./= norm(ψt)
    return copy(ψL), ψt
end

# ============================================================================
# Driver
# ============================================================================

"""
Run N-pair GRAPE.  Default random seed / initial-control recipe matches
2d_lattice_GRAPE.jl, so N=1 reproduces that script bit-for-bit.
"""
function run_npair(N::Int=1;
                   T=2π, num_steps=201, seed=42, max_iter=500,
                   save=true, psi0=nothing, psi_target=nothing, verbose=true)
    println("="^60)
    println("GRAPE (Trotter) — 2D Lattice, N = $N pair(s)")
    println("="^60)

    sys = build_system(N)
    @printf("Sector dims: Da = %d, Db = %d, D = %d\n", sys.Da, sys.Db, sys.D)

    if isnothing(psi0) || isnothing(psi_target)
        psi0_def, ψt_def = default_states(sys)
        psi0       = something(psi0, psi0_def)
        psi_target = something(psi_target, ψt_def)
    end

    dt = T / (num_steps - 1)
    @printf("Time: T = %.4f,  steps = %d,  dt = %.4f\n\n", T, num_steps, dt)

    Random.seed!(seed)
    t_arr = collect(range(0, T, length=num_steps))
    ctrls0 = zeros(num_steps, 9)
    for k in 1:6
        ctrls0[:, k] .= 0.1 * randn(num_steps)
    end
    ctrls0[:, 7] .= 1.0 .+ 0.1 * randn(num_steps)
    ctrls0[:, 8] .= 1.0 .+ 0.2*sin.(2π.*t_arr./T) .+ 0.1*randn(num_steps)
    ctrls0[:, 9] .= 1.0 .+ 0.2*cos.(2π.*t_arr./T) .+ 0.1*randn(num_steps)

    println("Starting GRAPE optimization...")
    println("-"^60)
    ctrls_opt, final_fidelity = grape_2d_Npair(
        sys, psi0, psi_target, T, num_steps;
        ctrls0=ctrls0, max_iter=max_iter, tol=1e-4, verbose=verbose
    )
    println("-"^60)
    @printf("Final fidelity: %.8f\n", final_fidelity)

    ψf = trotter_fwd(sys, Vector{ComplexF64}(psi0), ctrls_opt, dt)
    @printf("|⟨ψ_target|ψ_final⟩|² = %.8f\n", abs2(dot(psi_target, ψf)))

    if save
        output_file = "data/GRAPE_2d_Npair_$(N).jld2"
        println("\nSaving controls to $output_file ...")
        n = num_steps
        Va1, Va2, Va3 = ctrls_opt[:,1], ctrls_opt[:,2], ctrls_opt[:,3]
        Vb1, Vb2, Vb3 = ctrls_opt[:,4], ctrls_opt[:,5], ctrls_opt[:,6]
        U,   Ja,  Jb  = ctrls_opt[:,7], ctrls_opt[:,8], ctrls_opt[:,9]
        fidelity = final_fidelity
        Npair = N
        @save output_file Npair n T dt Va1 Va2 Va3 Vb1 Vb2 Vb3 U Ja Jb fidelity
    end

    return ctrls_opt, final_fidelity, sys
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 1
    run_npair(N)
end
