using LinearAlgebra, Random, Statistics
using Optim
using JLD2
using Printf

include("soft_boson.jl")

# ─────────────────────────────────────────────────────────────────────────────
# MPS GRAPE for 2D soft-core lattice, N a-b pairs
# ─────────────────────────────────────────────────────────────────────────────

"""
Run GRAPE optimization for N a-b pairs on the 4-site ring.

Saves the optimized pulse to data/GRAPE_2d_MPS_\$(N).jld2 in the same key
format as the exact-GRAPE pulse files (n, T, dt, Va1..Jb, fidelity, Npair).
"""
function run_grape(N::Int;
                   T::Float64   = 2π,
                   nsteps::Int  = 100,
                   cutoff       = 1e-10,
                   maxdim       = 64,
                   max_iter     = 100,
                   seed::Int    = 42,
                   save::Bool   = true,
                   verbose::Bool = true)

    dt = T / nsteps
    verbose && println("="^70)
    verbose && println("MPS GRAPE — N = $N pair(s),  T=$(round(T,digits=4)),  nsteps=$nsteps,  dt=$(round(dt,digits=5))")
    verbose && println("="^70)

    # ── Sites and states ─────────────────────────────────────────────────────
    s = siteinds("SoftBoson", 4)
    psi0, psi_target = default_states(s, N; cutoff, maxdim)

    # Precomputed per-site coefficients for diagonal-control gradients
    dVa1 = Float64[va_coeff(j,1,0,0) for j in 1:4]
    dVa2 = Float64[va_coeff(j,0,1,0) for j in 1:4]
    dVa3 = Float64[va_coeff(j,0,0,1) for j in 1:4]
    dVb1 = Float64[vb_coeff(j,1,0,0) for j in 1:4]
    dVb2 = Float64[vb_coeff(j,0,1,0) for j in 1:4]
    dVb3 = Float64[vb_coeff(j,0,0,1) for j in 1:4]
    dU   = ones(Float64, 4)

    # ── Single-site ⟨chi|O_j|psi⟩ ────────────────────────────────────────────
    function ess(chi, psi, O, j)
        p2 = copy(psi); p2[j] = noprime(O * p2[j])
        inner(chi, p2)
    end

    # ── ⟨chi | H_hop (unit amplitude) | psi⟩ using apply to preserve QN ──────
    H_up_tensors = hop_generator_tensors(s, :up)
    H_dn_tensors = hop_generator_tensors(s, :dn)
    function hop_inner_a(chi, psi)
        r = 0.0im
        for H in H_up_tensors
            r += inner(chi, apply([H], psi; cutoff=0, maxdim=4*maxdim))
        end
        return r
    end
    function hop_inner_b(chi, psi)
        r = 0.0im
        for H in H_dn_tensors
            r += inner(chi, apply([H], psi; cutoff=0, maxdim=4*maxdim))
        end
        return r
    end

    # ── Forward pass (stores 4 MPS per step for gradient) ───────────────────
    function forward(ctrls)
        phi1 = Vector{MPS}(undef, nsteps+1)
        phi2 = Vector{MPS}(undef, nsteps)
        phi3 = Vector{MPS}(undef, nsteps)
        phi4 = Vector{MPS}(undef, nsteps)
        psi = copy(psi0); phi1[1] = copy(psi)
        for n in 1:nsteps
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
            d1h = make_onsite_gates(s, Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
            psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi); phi2[n] = copy(psi)
            psi = apply(hop_gates_a(s, Ja, dt), psi; cutoff, maxdim); normalize!(psi); phi3[n] = copy(psi)
            psi = apply(hop_gates_b(s, Jb, dt), psi; cutoff, maxdim); normalize!(psi); phi4[n] = copy(psi)
            psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi); phi1[n+1] = copy(psi)
        end
        return phi1, phi2, phi3, phi4
    end

    function fidelity_only(ctrls)
        psi = copy(psi0)
        for n in 1:nsteps
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
            d1h = make_onsite_gates(s, Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
            psi = apply(vcat(d1h, hop_gates_a(s,Ja,dt), hop_gates_b(s,Jb,dt), d1h),
                        psi; cutoff, maxdim)
            normalize!(psi)
        end
        abs2(inner(psi_target, psi))
    end

    # ── Combined fidelity + gradient ─────────────────────────────────────────
    function compute_fg(ctrls)
        phi1, phi2, phi3, phi4 = forward(ctrls)
        overlap = inner(psi_target, phi1[nsteps+1])
        F = abs2(overlap)

        chi  = copy(psi_target)
        grad = zeros(Float64, nsteps, 9)

        for n in nsteps:-1:1
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]

            d1h_adj = make_onsite_gates(s, Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2; sign=-1)
            lambda_R = apply(d1h_adj, chi; cutoff, maxdim); normalize!(lambda_R)

            grad[n,9] = 2*real(conj(overlap) * (-im*dt) * hop_inner_b(lambda_R, phi4[n]))

            lambda_Jb = apply(hop_gates_b(s, Jb, dt; sign=-1), lambda_R; cutoff, maxdim)
            normalize!(lambda_Jb)

            grad[n,8] = 2*real(conj(overlap) * (-im*dt) * hop_inner_a(lambda_Jb, phi3[n]))

            lambda_Ja = apply(hop_gates_a(s, Ja, dt; sign=-1), lambda_Jb; cutoff, maxdim)
            normalize!(lambda_Ja)

            phi1_np1 = phi1[n+1]; phi2_n = phi2[n]
            nup_rr  = ComplexF64[ess(chi,      phi1_np1, op("Nup",  s[j]), j) for j in 1:4]
            ndn_rr  = ComplexF64[ess(chi,      phi1_np1, op("Ndn",  s[j]), j) for j in 1:4]
            ndou_rr = ComplexF64[ess(chi,      phi1_np1, op("Nupdn",s[j]), j) for j in 1:4]
            nup_ll  = ComplexF64[ess(lambda_Ja, phi2_n,  op("Nup",  s[j]), j) for j in 1:4]
            ndn_ll  = ComplexF64[ess(lambda_Ja, phi2_n,  op("Ndn",  s[j]), j) for j in 1:4]
            ndou_ll = ComplexF64[ess(lambda_Ja, phi2_n,  op("Nupdn",s[j]), j) for j in 1:4]

            f_d = conj(overlap) * (-im * dt/2)
            grad[n,1] = 2*real(f_d * dot(dVa1, nup_rr  .+ nup_ll))
            grad[n,2] = 2*real(f_d * dot(dVa2, nup_rr  .+ nup_ll))
            grad[n,3] = 2*real(f_d * dot(dVa3, nup_rr  .+ nup_ll))
            grad[n,4] = 2*real(f_d * dot(dVb1, ndn_rr  .+ ndn_ll))
            grad[n,5] = 2*real(f_d * dot(dVb2, ndn_rr  .+ ndn_ll))
            grad[n,6] = 2*real(f_d * dot(dVb3, ndn_rr  .+ ndn_ll))
            grad[n,7] = 2*real(f_d * dot(dU,   ndou_rr .+ ndou_ll))

            chi = apply(d1h_adj, lambda_Ja; cutoff, maxdim); normalize!(chi)
        end
        return F, grad
    end

    # ── L-BFGS driver ─────────────────────────────────────────────────────────
    iter    = Ref(0)
    t_opt   = Ref(time())
    F_cache = Ref(0.0)
    F_history = Float64[]

    function grad!(G, x)
        c = reshape(x, nsteps, 9)
        F, grd = compute_fg(c)
        F_cache[] = F; push!(F_history, F)
        G .= -vec(grd)
        iter[] += 1
        if verbose && (iter[] == 1 || iter[] % 5 == 0)
            elapsed = round(time() - t_opt[], digits=1)
            println("  iter $(lpad(iter[],4)) | F = $(lpad(round(F,digits=6),10)) | 1-F = $(lpad(round(1-F,sigdigits=3),9)) | t = $(elapsed)s")
            flush(stdout)
        end
    end
    loss(_) = 1.0 - F_cache[]

    Random.seed!(seed)
    c0 = 0.3 .* randn(nsteps, 9)
    c0[:,7] .+= 1.0; c0[:,8] .+= 1.0; c0[:,9] .+= 1.0

    g0 = zeros(nsteps*9); grad!(g0, vec(c0))
    t_opt[] = time()
    verbose && println("Starting L-BFGS (m=30, max_iter=$max_iter)..."); flush(stdout)

    result = Optim.optimize(
        loss, grad!, vec(c0),
        LBFGS(m=30),
        Optim.Options(
            iterations  = max_iter,
            g_tol       = 1e-6,
            f_reltol    = 1e-10,
            show_trace  = false,
            store_trace = true,
            callback    = state -> begin
                F_cur = 1.0 - state.f_x
                if F_cur > 0.99
                    verbose && println("  Early stop: F = $(round(F_cur,digits=6)) > 0.99")
                    return true
                end
                return false
            end,
        )
    )

    c_opt   = reshape(Optim.minimizer(result), nsteps, 9)
    F_opt   = 1.0 - Optim.minimum(result)
    F_check = fidelity_only(c_opt)

    if verbose
        println("═"^70)
        println("  N=$N | grad evals=$(iter[]) | F(optim)=$(round(F_opt,digits=6)) | F(forward)=$(round(F_check,digits=6))")
        println("  1-F = $(round(1-F_check,sigdigits=4)) | converged=$(Optim.converged(result)) | t=$(round(time()-t_opt[],digits=1))s")
    end

    # ── Save pulse in the same format as exact_GRAPE.jl output ───────────────
    if save
        # Pad arrays to length n = nsteps+1 (MPS_sim/exact_sim convention)
        pad(v) = vcat(v, zero(eltype(v)))
        n_save = nsteps + 1
        output_file = "data/GRAPE_2d_MPS_$(N).jld2"
        jldsave(output_file;
            Npair = N, n = n_save, T = T, dt = dt,
            Va1 = pad(c_opt[:,1]), Va2 = pad(c_opt[:,2]), Va3 = pad(c_opt[:,3]),
            Vb1 = pad(c_opt[:,4]), Vb2 = pad(c_opt[:,5]), Vb3 = pad(c_opt[:,6]),
            U   = pad(c_opt[:,7]), Ja  = pad(c_opt[:,8]), Jb  = pad(c_opt[:,9]),
            fidelity = F_check)
        verbose && println("  Saved $output_file")
    end

    return (controls=c_opt, fidelity=F_check, F_history=F_history, result=result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = Dict{Int,Any}()
    for N in 1:3
        results[N] = run_grape(N)
    end
    println()
    println("─── Summary ────────────────────────────────────────────")
    for N in 1:3
        @printf("  N=%d : F = %.6f\n", N, results[N].fidelity)
    end
end
