using Printf, Random, LinearAlgebra, JLD2

include("exact_GRAPE.jl")

# Dense forward time evolution under a seeded random time-varying pulse, for
# N = 1..5 pairs. T = 2π, nsteps = 100.

const T      = 2π
const NSTEPS = 100
const DT     = T / NSTEPS
const SEED   = 42

function make_pulse(nsteps::Int, T::Float64; seed::Int=42)
    Random.seed!(seed)
    t_arr = collect(range(0, T, length=nsteps))
    c = zeros(nsteps, 9)
    for k in 1:6; c[:, k] .= 0.1 * randn(nsteps); end
    c[:, 7] .= 1.0 .+ 0.1 * randn(nsteps)
    c[:, 8] .= 1.0 .+ 0.2 * sin.(2π .* t_arr ./ T) .+ 0.1 * randn(nsteps)
    c[:, 9] .= 1.0 .+ 0.2 * cos.(2π .* t_arr ./ T) .+ 0.1 * randn(nsteps)
    return c
end

"""
Factor-exploiting Trotter step.

Instead of eigendecomposing the D×D matrices H_Ja_mat = h_a⊗I_b and
H_Jb_mat = I_a⊗h_b, we eigendecompose the two small Da×Da / Db×Db factors
and apply exp(-iJτh_a)⊗I via reshape + matmul. This turns the O(D³)
eigendecomp into O(Da³), crucial past N=5 where D²≥10⁸.

H₁_diag = Va·dVa + … + U·dU  (D-vector, already cached in `sys`).
"""
function trotter_step_factored!(psi::Vector{ComplexF64}, sys,
                                Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt,
                                evals_a, evecs_a, evals_b, evecs_b)
    Da, Db = sys.Da, sys.Db
    # Diagonal half-step
    h = Va1 .* sys.dVa[1] .+ Va2 .* sys.dVa[2] .+ Va3 .* sys.dVa[3] .+
        Vb1 .* sys.dVb[1] .+ Vb2 .* sys.dVb[2] .+ Vb3 .* sys.dVb[3] .+
        U   .* sys.dU
    d1 = exp.(-im * (dt/2) .* h)
    psi .= d1 .* psi

    # Flat index: (ai-1)*Db + bi  ⇒  column-major reshape is (Db, Da),
    # giving M[bi, ai] = psi[(ai-1)*Db + bi].
    #   kron(h_a, I_b) · ψ  ⇔  M → M * h_a
    #   kron(I_a, h_b) · ψ  ⇔  M → h_b * M
    M = reshape(psi, Db, Da)

    pa = exp.(-im * Ja * dt .* evals_a)                           # Da-vec
    M_new = ((M * evecs_a) .* transpose(pa)) * evecs_a'            # right-multiply by exp(-iJadth_a)

    pb = exp.(-im * Jb * dt .* evals_b)                           # Db-vec
    M_new = evecs_b * (pb .* (evecs_b' * M_new))                   # left-multiply by exp(-iJbdth_b)

    psi .= d1 .* vec(M_new)
    return psi
end

"""
Forward-propagate ψ₀ through nsteps using the factor-exploiting step.
Return the fidelity trajectory |⟨ψ_T | ψ(t)⟩|² of length nsteps+1.
"""
function evolve!(sys, psi0, psi_target, ctrls, dt)
    # Rebuild the SMALL Da×Da / Db×Db hopping factors directly from the basis,
    # bypassing the D×D Kronecker matrices (which would need an O(D³) eigendecomp).
    h_a = species_hopping(sys.basis_a)
    h_b = species_hopping(sys.basis_b)
    Fa = eigen(Hermitian(Matrix(h_a))); evals_a = Fa.values; evecs_a = Matrix{ComplexF64}(Fa.vectors)
    Fb = eigen(Hermitian(Matrix(h_b))); evals_b = Fb.values; evecs_b = Matrix{ComplexF64}(Fb.vectors)

    nsteps = size(ctrls, 1)
    fid = Vector{Float64}(undef, nsteps + 1)
    psi = Vector{ComplexF64}(psi0)
    fid[1] = abs2(dot(psi_target, psi))
    for n in 1:nsteps
        Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb = ctrls[n, :]
        trotter_step_factored!(psi, sys,
                               Va1, Va2, Va3, Vb1, Vb2, Vb3, U, Ja, Jb, dt,
                               evals_a, evecs_a, evals_b, evecs_b)
        fid[n + 1] = abs2(dot(psi_target, psi))
    end
    return fid, psi
end

max_N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 5

# Lightweight system builder: same content as build_system(N) in exact_GRAPE.jl
# but skips the D×D H_Ja_mat = kron(h_a, I_b) and H_Jb_mat = kron(I_a, h_b).
# At N=10, D² = 81796² complex = 107 GB each — that's the OOM. The factored
# step only needs the Da×Da / Db×Db hopping factors, so don't build the Kron.
struct FactoredSys
    N::Int
    Da::Int
    Db::Int
    D::Int
    basis_a::Vector{Vector{Int}}
    basis_b::Vector{Vector{Int}}
    dVa::Vector{Vector{Float64}}
    dVb::Vector{Vector{Float64}}
    dU::Vector{Float64}
end

function build_system_factored(N::Int)
    @assert N ≥ 1
    basis_a = species_basis(L_SITES, N)
    basis_b = species_basis(L_SITES, N)
    Da = length(basis_a); Db = length(basis_b); D = Da * Db

    n_a = [species_occupation(basis_a, j) for j in 0:L_SITES-1]
    n_b = [species_occupation(basis_b, j) for j in 0:L_SITES-1]

    # Diagonal V_k potentials (same convention as build_system).
    dVa = [kron(n_a[k+1] .- n_a[1], ones(Db)) for k in 1:3]
    dVb = [kron(ones(Da), n_b[k+1] .- n_b[1]) for k in 1:3]

    dU = zeros(Float64, D)
    for j in 1:L_SITES
        dU .+= kron(n_a[j], n_b[j])
    end

    return FactoredSys(N, Da, Db, D, basis_a, basis_b, dVa, dVb, dU)
end
ctrls = make_pulse(NSTEPS, T; seed=SEED)
println("━"^74)
println("Dense time evolution — T=$(round(T,digits=4)), nsteps=$NSTEPS, dt=$(round(DT,digits=5)), seed=$SEED, N=1..$max_N")
println("━"^74)
@printf("%3s  %6s  %6s  %6s  %10s  %10s  %10s\n",
        "N", "D_a", "D_b", "D", "F(0)", "F(T)", "t_wall(s)")

results = Dict{Int, Any}()
for N in 1:max_N
    sys = build_system_factored(N)
    psi0 = zeros(ComplexF64, sys.D)
    psi0[(species_index(sys.basis_a, fill(0, N)) - 1) * sys.Db +
          species_index(sys.basis_b, fill(1, N))] = 1.0
    psi_R = zeros(ComplexF64, sys.D)
    psi_R[(species_index(sys.basis_a, fill(2, N)) - 1) * sys.Db +
          species_index(sys.basis_b, fill(3, N))] = 1.0
    psi_target = (psi0 .+ psi_R) ./ sqrt(2.0)

    # JIT warmup (discard timing)
    _ = evolve!(sys, psi0, psi_target, ctrls, DT)

    t0 = time()
    fid, psi_f = evolve!(sys, psi0, psi_target, ctrls, DT)
    dt = time() - t0

    @printf("%3d  %6d  %6d  %6d  %10.6f  %10.6f  %10.3f\n",
            N, sys.Da, sys.Db, sys.D, fid[1], fid[end], dt)
    results[N] = (N=N, Da=sys.Da, Db=sys.Db, D=sys.D,
                  fid=fid, F_final=fid[end], wall=dt)
end
println("━"^74)

# Save results for later inspection / plotting.
out_dir = joinpath(@__DIR__, "data")
mkpath(out_dir)
out_file = joinpath(out_dir, "dense_time_evol_seed$(SEED)_N$(max_N).jld2")
jldsave(out_file;
    T = T, nsteps = NSTEPS, dt = DT, seed = SEED,
    ctrls = ctrls,
    Ns = collect(1:max_N),
    D  = [results[N].D      for N in 1:max_N],
    fid_trajs = [results[N].fid for N in 1:max_N],
    F_final   = [results[N].F_final for N in 1:max_N],
    wall      = [results[N].wall for N in 1:max_N])
println("Saved $out_file")
