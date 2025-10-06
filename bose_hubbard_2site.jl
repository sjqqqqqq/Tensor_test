using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 2
    N_particles = 100

    # Time-dependent parameters
    J(t) = 1.0 + 0.5*sin(0.5*t)
    U(t) = 1.0 * cos(0.5 * t)
    Δ(t) = -0.1 * t

    # Time evolution parameters
    tau = 0.01
    ttotal = 10.0
    cutoff = 1E-8

    # Create site indices
    s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

    # Helper function to build time-dependent Hamiltonian MPO
    function make_H(J_val, U_val, Δ_val, s)
        os = OpSum()
        # Hopping terms
        os += -J_val, "Adag", 1, "A", 2
        os += -J_val, "A", 1, "Adag", 2
        # On-site interaction terms
        for j in 1:N_sites
            os += U_val/2, "N * N", j
            os += -U_val/2, "N", j
        end
        # Tilt terms
        for j in 1:N_sites
            os += (j-1)*Δ_val, "N", j
        end
        return MPO(os, s)
    end

    # Initial state
    psi = MPS(s, ["$N_particles", "0"])

    println("Time\tJ(t)\tU(t)\tΔ(t)\tSite1\tSite2\tTotal")

    # Time evolution using TDVP
    t = 0.0
    while t <= ttotal
        n1, n2 = expect(psi, "N"; sites=1), expect(psi, "N"; sites=2)
        N1, N2 = expect(psi, "N * N"; sites=1), expect(psi, "N * N"; sites=2)
        sz1, sz2 = 
        J_t, U_t, Δ_t= J(t), U(t), Δ(t)
        println("$t\t$(round(J_t, digits=3))\t$(round(U_t, digits=3))\t$(round(Δ_t, digits=3))\t$(round(n1, digits=2))\t$(round(n2, digits=2))\t$(round(n1+n2, digits=2))")

        t ≈ ttotal && break

        # Build Hamiltonian at current time
        H = make_H(J_t, U_t, Δ_t, s)

        # Evolve using TDVP for one time step
        psi = tdvp(H, -im * tau, psi; cutoff, normalize=true, nsite=2)

        t += tau
    end
end
