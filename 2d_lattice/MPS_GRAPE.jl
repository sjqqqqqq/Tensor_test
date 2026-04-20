using ITensors, ITensorMPS
using LinearAlgebra, Random, Statistics
using Optim
using JLD2

# ── Soft-core two-species boson site type ─────────────────────────────────────
# Local basis: |nₐ, n_b⟩  with  nₐ, n_b ∈ 0..NMAX
# Index ordering: i = nₐ*(NMAX+1) + n_b + 1  (1-based)
# States: |0,0⟩=1, |0,1⟩=2, |0,2⟩=3, |1,0⟩=4, |1,1⟩=5, ...
const NMAX = 2   # max occupancy per species per site

ITensors.space(::SiteType"SoftBoson") = (NMAX+1)^2

function ITensors.state(::StateName"Emp",  ::SiteType"SoftBoson", s::Index)
    T = ITensor(s); T[s=>1]      = 1.0; return T   # |0,0⟩
end
function ITensors.state(::StateName"Dn",   ::SiteType"SoftBoson", s::Index)
    T = ITensor(s); T[s=>2]      = 1.0; return T   # |0,1⟩
end
function ITensors.state(::StateName"Up",   ::SiteType"SoftBoson", s::Index)
    T = ITensor(s); T[s=>NMAX+2] = 1.0; return T   # |1,0⟩
end
function ITensors.state(::StateName"UpDn", ::SiteType"SoftBoson", s::Index)
    T = ITensor(s); T[s=>NMAX+3] = 1.0; return T   # |1,1⟩
end

function ITensors.op(::OpName"Nup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(na); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Ndn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(nb); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Nupdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX; i = na*d+nb+1; mat[i,i] = Float64(na*nb); end
    return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdagup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX-1, nb in 0:NMAX
        i = na*d+nb+1; ip = (na+1)*d+nb+1; mat[ip,i] = sqrt(Float64(na+1))
    end; return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cup", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 1:NMAX, nb in 0:NMAX
        i = na*d+nb+1; im_ = (na-1)*d+nb+1; mat[im_,i] = sqrt(Float64(na))
    end; return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdagdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 0:NMAX-1
        i = na*d+nb+1; ip = na*d+nb+2; mat[ip,i] = sqrt(Float64(nb+1))
    end; return ITensor(mat, s', dag(s))
end
function ITensors.op(::OpName"Cdn", ::SiteType"SoftBoson", s::Index)
    d = NMAX+1; dim2 = d^2; mat = zeros(dim2, dim2)
    for na in 0:NMAX, nb in 1:NMAX
        i = na*d+nb+1; im_ = na*d+nb; mat[im_,i] = sqrt(Float64(nb))
    end; return ITensor(mat, s', dag(s))
end

let
    # ── System parameters ─────────────────────────────────────────────────────
    M      = 1        # number of a–b pairs
    T      = 2π
    nsteps = 100      # Trotter steps; dt = T / nsteps
    dt     = T / nsteps
    cutoff = 1e-10
    maxdim = 64

    # ── 4-site ring topology (Gamma4, 0-indexed → 1-indexed ITensor) ──────────
    # Bonds 0-indexed: (0,1),(0,2),(1,3),(2,3)  →  Group A adjacent, Group B non-adjacent
    bonds_A = [(1,2), (3,4)]
    bonds_B = [(1,3), (2,4)]
    all_bonds = vcat(bonds_A, bonds_B)

    # ── Site indices ──────────────────────────────────────────────────────────
    # "SoftBoson": local basis |nₐ,n_b⟩, nₐ,n_b ∈ 0..NMAX  (dim = (NMAX+1)² = 9)
    s = siteinds("SoftBoson", 4)

    # ── States ────────────────────────────────────────────────────────────────
    psi0   = MPS(s, ["Up","Dn","Emp","Emp"])    # |a@0, b@1⟩
    psi_t1 = MPS(s, ["Up","Dn","Emp","Emp"])
    psi_t2 = MPS(s, ["Emp","Emp","Up","Dn"])
    psi_target = normalize!(add(psi_t1, psi_t2; cutoff, maxdim))

    # ── On-site coefficient helpers ───────────────────────────────────────────
    # H1 = Va1(n^a_1−n^a_0)+Va2(n^a_2−n^a_0)+Va3(n^a_3−n^a_0)
    #     +Vb1(n^b_1−n^b_0)+Vb2(n^b_2−n^b_0)+Vb3(n^b_3−n^b_0)  + U·Σ_j n^a_j n^b_j
    va_coeff(j,Va1,Va2,Va3) = j==1 ? -(Va1+Va2+Va3) : j==2 ? Va1 : j==3 ? Va2 : Va3
    vb_coeff(j,Vb1,Vb2,Vb3) = j==1 ? -(Vb1+Vb2+Vb3) : j==2 ? Vb1 : j==3 ? Vb2 : Vb3

    # Precomputed coefficient vectors for partial derivatives (length-4, 1-indexed sites)
    dVa1 = Float64[va_coeff(j,1,0,0) for j in 1:4]   # [-1,1,0,0]
    dVa2 = Float64[va_coeff(j,0,1,0) for j in 1:4]   # [-1,0,1,0]
    dVa3 = Float64[va_coeff(j,0,0,1) for j in 1:4]   # [-1,0,0,1]
    dVb1 = Float64[vb_coeff(j,1,0,0) for j in 1:4]
    dVb2 = Float64[vb_coeff(j,0,1,0) for j in 1:4]
    dVb3 = Float64[vb_coeff(j,0,0,1) for j in 1:4]
    dU   = ones(Float64, 4)

    # ── Gate factories ────────────────────────────────────────────────────────
    function make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, τ, sign=1)
        [exp(sign*(-im)*τ*(va_coeff(j,Va1,Va2,Va3)*op("Nup",s[j]) +
                           vb_coeff(j,Vb1,Vb2,Vb3)*op("Ndn",s[j]) +
                           U*op("Nupdn",s[j]))) for j in 1:4]
    end
    # sign=+1 → forward (exp(-im*τ*H)), sign=-1 → adjoint (exp(+im*τ*H))
    make_adj_onsite(Va1,Va2,Va3,Vb1,Vb2,Vb3,U,τ) = make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U,τ,-1)

    make_hop_a(j,k,Ja,τ) = exp(-im*τ*Ja*(op("Cdagup",s[j])*op("Cup",   s[k]) +
                                          op("Cup",   s[j])*op("Cdagup",s[k])))
    make_hop_b(j,k,Jb,τ) = exp(-im*τ*Jb*(op("Cdagdn",s[j])*op("Cdn",   s[k]) +
                                          op("Cdn",   s[j])*op("Cdagdn",s[k])))
    make_adj_hop_a(j,k,Ja,τ) = exp(+im*τ*Ja*(op("Cdagup",s[j])*op("Cup",   s[k]) +
                                              op("Cup",   s[j])*op("Cdagup",s[k])))
    make_adj_hop_b(j,k,Jb,τ) = exp(+im*τ*Jb*(op("Cdagdn",s[j])*op("Cdn",   s[k]) +
                                              op("Cdn",   s[j])*op("Cdagdn",s[k])))

    # Full set of hopping gates for one species (2nd-order sub-Trotter: A/2→B→A/2)
    hop_gates_a(Ja,τ) = vcat([make_hop_a(j,k,Ja,τ/2) for (j,k) in bonds_A],
                              [make_hop_a(j,k,Ja,τ)   for (j,k) in bonds_B],
                              [make_hop_a(j,k,Ja,τ/2) for (j,k) in bonds_A])
    hop_gates_b(Jb,τ) = vcat([make_hop_b(j,k,Jb,τ/2) for (j,k) in bonds_A],
                              [make_hop_b(j,k,Jb,τ)   for (j,k) in bonds_B],
                              [make_hop_b(j,k,Jb,τ/2) for (j,k) in bonds_A])
    adj_hop_gates_a(Ja,τ) = vcat([make_adj_hop_a(j,k,Ja,τ/2) for (j,k) in bonds_A],
                                   [make_adj_hop_a(j,k,Ja,τ)   for (j,k) in bonds_B],
                                   [make_adj_hop_a(j,k,Ja,τ/2) for (j,k) in bonds_A])
    adj_hop_gates_b(Jb,τ) = vcat([make_adj_hop_b(j,k,Jb,τ/2) for (j,k) in bonds_A],
                                   [make_adj_hop_b(j,k,Jb,τ)   for (j,k) in bonds_B],
                                   [make_adj_hop_b(j,k,Jb,τ/2) for (j,k) in bonds_A])

    # ── Expectation-value helpers ─────────────────────────────────────────────
    # ⟨chi|O_j|psi⟩ for single-site operator at site j
    function ess(chi, psi, O, j)
        p2 = copy(psi); p2[j] = noprime(O * p2[j])
        inner(chi, p2)
    end

    # ⟨chi | Σ_bonds (c†^a_j c^a_k + c^a_j c†^a_k) | psi⟩  (unit-Ja hopping)
    # Uses apply([h_jk], psi; cutoff=0) to preserve QN block structure correctly.
    function hop_inner_a(chi, psi)
        r = 0.0im
        for (j,k) in all_bonds
            h_jk = op("Cdagup",s[j])*op("Cup",   s[k]) +
                   op("Cup",   s[j])*op("Cdagup",s[k])
            psi_h = apply([h_jk], psi; cutoff=0, maxdim=4*maxdim)
            r += inner(chi, psi_h)
        end
        return r
    end

    # ⟨chi | Σ_bonds (c†^b_j c^b_k + c^b_j c†^b_k) | psi⟩  (unit-Jb hopping)
    function hop_inner_b(chi, psi)
        r = 0.0im
        for (j,k) in all_bonds
            h_jk = op("Cdagdn",s[j])*op("Cdn",   s[k]) +
                   op("Cdn",   s[j])*op("Cdagdn",s[k])
            psi_h = apply([h_jk], psi; cutoff=0, maxdim=4*maxdim)
            r += inner(chi, psi_h)
        end
        return r
    end

    # ── Forward pass — stores 4 MPS per step for gradient computation ─────────
    #
    #  Step n:  phi1[n] →(D1h)→ phi2[n] →(U_Ja)→ phi3[n] →(U_Jb)→ phi4[n] →(D1h)→ phi1[n+1]
    #
    function forward(ctrls)
        phi1 = Vector{MPS}(undef, nsteps+1)   # state entering step n (and final state)
        phi2 = Vector{MPS}(undef, nsteps)      # after left D1_half
        phi3 = Vector{MPS}(undef, nsteps)      # after U_Ja
        phi4 = Vector{MPS}(undef, nsteps)      # after U_Jb

        psi = copy(psi0)
        phi1[1] = copy(psi)

        for n in 1:nsteps
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
            d1h = make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)

            psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi)
            phi2[n] = copy(psi)

            psi = apply(hop_gates_a(Ja,dt), psi; cutoff, maxdim); normalize!(psi)
            phi3[n] = copy(psi)

            psi = apply(hop_gates_b(Jb,dt), psi; cutoff, maxdim); normalize!(psi)
            phi4[n] = copy(psi)

            psi = apply(d1h, psi; cutoff, maxdim); normalize!(psi)
            phi1[n+1] = copy(psi)
        end

        return phi1, phi2, phi3, phi4
    end

    # ── Fidelity only (no storage; used for finite-difference check) ──────────
    function fidelity_only(ctrls)
        psi = copy(psi0)
        for n in 1:nsteps
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]
            d1h = make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
            psi = apply(vcat(d1h, hop_gates_a(Ja,dt), hop_gates_b(Jb,dt), d1h),
                        psi; cutoff, maxdim)
            normalize!(psi)
        end
        abs2(inner(psi_target, psi))
    end

    # ── Fidelity + gradient (combined forward–backward GRAPE sweep) ───────────
    #
    # Backward step n (given chi = χ_{n+1}):
    #   lambda_R  = D1h†      · chi       → gradient for Jb
    #   lambda_Jb = U_Jb†     · lambda_R  → gradient for Ja
    #   lambda_Ja = U_Ja†     · lambda_Jb → gradient for Va/Vb/U (left D1h)
    #   chi_new   = D1h†      · lambda_Ja → χ_n (propagated back one step)
    #
    # Gradient formulas (factor_h = conj(ov)*(-im*dt)):
    #   grad_Ja[n]  = 2Re[factor_h   · ⟨lambda_Jb | H_Ja_unit | phi3[n]⟩]
    #   grad_Jb[n]  = 2Re[factor_h   · ⟨lambda_R  | H_Jb_unit | phi4[n]⟩]
    #   grad_ck[n]  = 2Re[factor_h/2 · (⟨chi|∂H1/∂ck|phi1[n+1]⟩ + ⟨lambda_Ja|∂H1/∂ck|phi2[n]⟩)]
    #
    function compute_fg(ctrls)
        phi1, phi2, phi3, phi4 = forward(ctrls)
        overlap = inner(psi_target, phi1[nsteps+1])
        F = abs2(overlap)

        chi  = copy(psi_target)
        grad = zeros(Float64, nsteps, 9)

        for n in nsteps:-1:1
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = ctrls[n,:]

            # ── Step 1: lambda_R = D1h† · chi ─────────────────────────────────
            d1h_adj = make_adj_onsite(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
            lambda_R = apply(d1h_adj, chi; cutoff, maxdim); normalize!(lambda_R)

            # ── Gradient for Jb ────────────────────────────────────────────────
            grad[n,9] = 2*real(conj(overlap) * (-im*dt) * hop_inner_b(lambda_R, phi4[n]))

            # ── Step 2: lambda_Jb = U_Jb† · lambda_R ─────────────────────────
            lambda_Jb = apply(adj_hop_gates_b(Jb,dt), lambda_R; cutoff, maxdim)
            normalize!(lambda_Jb)

            # ── Gradient for Ja ────────────────────────────────────────────────
            grad[n,8] = 2*real(conj(overlap) * (-im*dt) * hop_inner_a(lambda_Jb, phi3[n]))

            # ── Step 3: lambda_Ja = U_Ja† · lambda_Jb ────────────────────────
            lambda_Ja = apply(adj_hop_gates_a(Ja,dt), lambda_Jb; cutoff, maxdim)
            normalize!(lambda_Ja)

            # ── Gradients for diagonal controls (Va1..U) ──────────────────────
            # Precompute single-site inner products:
            #   rr[j] = ⟨chi       | O_j | phi1[n+1]⟩   (right D1h contribution)
            #   ll[j] = ⟨lambda_Ja | O_j | phi2[n]  ⟩   (left  D1h contribution)
            phi1_np1 = phi1[n+1]
            phi2_n   = phi2[n]
            nup_rr   = ComplexF64[ess(chi,      phi1_np1, op("Nup",  s[j]), j) for j in 1:4]
            ndn_rr   = ComplexF64[ess(chi,      phi1_np1, op("Ndn",  s[j]), j) for j in 1:4]
            ndou_rr  = ComplexF64[ess(chi,      phi1_np1, op("Nupdn",s[j]), j) for j in 1:4]
            nup_ll   = ComplexF64[ess(lambda_Ja, phi2_n,  op("Nup",  s[j]), j) for j in 1:4]
            ndn_ll   = ComplexF64[ess(lambda_Ja, phi2_n,  op("Ndn",  s[j]), j) for j in 1:4]
            ndou_ll  = ComplexF64[ess(lambda_Ja, phi2_n,  op("Nupdn",s[j]), j) for j in 1:4]

            f_d = conj(overlap) * (-im * dt/2)
            grad[n,1] = 2*real(f_d * dot(dVa1, nup_rr  .+ nup_ll))
            grad[n,2] = 2*real(f_d * dot(dVa2, nup_rr  .+ nup_ll))
            grad[n,3] = 2*real(f_d * dot(dVa3, nup_rr  .+ nup_ll))
            grad[n,4] = 2*real(f_d * dot(dVb1, ndn_rr  .+ ndn_ll))
            grad[n,5] = 2*real(f_d * dot(dVb2, ndn_rr  .+ ndn_ll))
            grad[n,6] = 2*real(f_d * dot(dVb3, ndn_rr  .+ ndn_ll))
            grad[n,7] = 2*real(f_d * dot(dU,   ndou_rr .+ ndou_ll))

            # ── Step 4: chi_new = D1h† · lambda_Ja  (= χ_n) ─────────────────
            chi = apply(d1h_adj, lambda_Ja; cutoff, maxdim); normalize!(chi)
        end

        return F, grad
    end

    # ── JIT warmup ────────────────────────────────────────────────────────────
    println("JIT warmup..."); flush(stdout)
    c_warm = zeros(nsteps, 9); c_warm[:,8] .= 0.5; c_warm[:,9] .= 0.5
    compute_fg(c_warm)
    fidelity_only(c_warm)
    println("Warmup done.\n"); flush(stdout)

    # ── Gradient accuracy check (analytic vs central finite difference) ───────
    println("Gradient check (analytic vs central-difference, selected components):"); flush(stdout)
    Random.seed!(1)
    c_test = 0.3 .* randn(nsteps, 9)
    c_test[:,8] .+= 1.0; c_test[:,9] .+= 1.0   # reasonable Ja, Jb baseline

    F0, ag = compute_fg(c_test)
    h_fd   = 1e-5
    println("  Reference fidelity F = $(round(F0, digits=6))"); flush(stdout)

    # Check 5 components: spread across control indices and time steps
    checks = [(1,1), (100,4), (50,7), (75,8), (90,9)]
    for (n_t, k_c) in checks
        cp = copy(c_test); cp[n_t,k_c] += h_fd
        cm = copy(c_test); cm[n_t,k_c] -= h_fd
        ng = (fidelity_only(cp) - fidelity_only(cm)) / (2*h_fd)
        err = abs(ag[n_t,k_c] - ng)
        lab = ["Va1","Va2","Va3","Vb1","Vb2","Vb3","U","Ja","Jb"][k_c]
        println("  c[$n_t,$lab]:  analytic=$(lpad(round(ag[n_t,k_c],sigdigits=5),11))  " *
                "numeric=$(lpad(round(ng,sigdigits=5),11))  err=$(round(err,sigdigits=2))")
        flush(stdout)
    end
    println(); flush(stdout)

    # ── GRAPE optimization (L-BFGS) ───────────────────────────────────────────
    iter    = Ref(0)
    t_opt   = Ref(time())
    F_cache = Ref(0.0)
    F_history = Float64[]

    function grad!(G, x)
        c = reshape(x, nsteps, 9)
        F, grd = compute_fg(c)
        F_cache[] = F
        push!(F_history, F)
        G .= -vec(grd)    # minimise 1-F → gradient of (1-F) = -gradient of F
        iter[] += 1
        elapsed = round(time() - t_opt[], digits=1)
        if iter[] == 1 || iter[] % 5 == 0
            println("  iter $(lpad(iter[],4)) | " *
                    "F = $(lpad(round(F,digits=6),10)) | " *
                    "1-F = $(lpad(round(1-F,sigdigits=3),9)) | " *
                    "t = $(elapsed)s")
            flush(stdout)
        end
    end

    loss(_) = 1.0 - F_cache[]

    Random.seed!(42)
    c0 = 0.3 .* randn(nsteps, 9)
    c0[:,7] .+= 1.0    # U  near 1
    c0[:,8] .+= 1.0    # Ja near 1
    c0[:,9] .+= 1.0    # Jb near 1

    # Initialise cache
    g0 = zeros(nsteps*9)
    grad!(g0, vec(c0))

    t_opt[] = time()
    println("Starting GRAPE optimization")
    println("  Model  : 4-site ring, M=$M pair(s), soft-core bosons (NMAX=$NMAX), T=$(round(T,digits=4)), nsteps=$nsteps")
    println("  Task   : |a@0,b@1⟩ → (|a@0,b@1⟩+|a@2,b@3⟩)/√2  (SPDC-like Bell pair)")
    println("  Method : L-BFGS (m=30), max 200 iterations")
    println(); flush(stdout)

    result = Optim.optimize(
        loss,
        grad!,
        vec(c0),
        LBFGS(m=30),
        Optim.Options(
            iterations  = 200,
            g_tol       = 1e-6,
            f_reltol    = 1e-10,
            show_trace  = false,
            store_trace = true,
            callback    = state -> begin
                F_cur = 1.0 - state.f_x
                if F_cur > 0.9999
                    println("  Early stop: F = $(round(F_cur,digits=6)) > 0.9999")
                    flush(stdout)
                    return true
                end
                return false
            end,
        )
    )

    c_opt   = reshape(Optim.minimizer(result), nsteps, 9)
    F_opt   = 1.0 - Optim.minimum(result)
    F_check = fidelity_only(c_opt)

    println()
    println("══════════════════════════════════════════════════════════")
    println("  Total gradient evals   = $(iter[])")
    println("  F from optimizer       = $(round(F_opt,   digits=6))")
    println("  F verification (fwd)   = $(round(F_check, digits=6))")
    println("  Infidelity 1-F         = $(round(1-F_check, sigdigits=4))")
    println("  Converged              = $(Optim.converged(result))")
    println("  Total time             = $(round(time()-t_opt[],digits=1))s")
    println()

    # ── Final state diagnostics ───────────────────────────────────────────────
    psi_opt = let psi = copy(psi0)
        for n in 1:nsteps
            Va1,Va2,Va3,Vb1,Vb2,Vb3,U,Ja,Jb = c_opt[n,:]
            d1h = make_onsite_gates(Va1,Va2,Va3,Vb1,Vb2,Vb3,U, dt/2)
            psi = apply(vcat(d1h, hop_gates_a(Ja,dt), hop_gates_b(Jb,dt), d1h),
                        psi; cutoff, maxdim)
            normalize!(psi)
        end
        psi
    end
    nup_final = real(expect(psi_opt, "Nup"))
    ndn_final = real(expect(psi_opt, "Ndn"))
    println("  n_up profile at t=T  : ", round.(nup_final, digits=3))
    println("  n_dn profile at t=T  : ", round.(ndn_final, digits=3))
    println("  Max bond dim         : ", maxlinkdim(psi_opt))
    println()
    println("  Control pulse stats (all 9 controls):")
    for (k,name) in enumerate(["Va1","Va2","Va3","Vb1","Vb2","Vb3","U","Ja","Jb"])
        println("    $name : max|c|=$(round(maximum(abs,c_opt[:,k]),digits=3))"*
                "  rms=$(round(sqrt(mean(c_opt[:,k].^2)),digits=3))")
    end

    # ── Save pulse (JLD2, same format as GRAPE_2d_pulse.jld2) ────────────────
    jldsave("data/GRAPE_2d_pulse_MPS.jld2";
        n0       = nsteps,
        dt       = dt,
        Va1      = c_opt[:,1],
        Va2      = c_opt[:,2],
        Va3      = c_opt[:,3],
        Vb1      = c_opt[:,4],
        Vb2      = c_opt[:,5],
        Vb3      = c_opt[:,6],
        UH       = c_opt[:,7],
        Ja       = c_opt[:,8],
        Jb       = c_opt[:,9],
        fidelity = F_check,
    )
    println("  Saved data/GRAPE_2d_pulse_MPS.jld2")

    (controls=c_opt, fidelity=F_check, F_history=F_history, result=result)
end
