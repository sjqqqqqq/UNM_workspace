# HEngg_GRAPE.jl
# Quantum control optimization using GRAPE.jl
# Solves the same problem as HEngg_ode.jl: state transfer from |0,1⟩ to (|0,1⟩ + |2,3⟩)/√2

using LinearAlgebra
using GRAPE
using QuantumControl
using QuantumControl.Functionals: J_T_sm
using QuantumControl.QuantumPropagators: hamiltonian
using JLD2
using Plots
using Random

# ===== QUANTUM SYSTEM SETUP (same as HEngg_ode.jl) =====
const Gamma = Float64[0 1 1 0;
                      1 0 0 1;
                      1 0 0 1;
                      0 1 1 0]

const I4 = Matrix{ComplexF64}(I, 4, 4)

function projector(site_idx)
    e = zeros(ComplexF64, 4, 1)
    e[site_idx + 1] = 1.0
    return e * e'
end

const P_a_sites = [kron(projector(s), I4) for s in 0:3]
const P_b_sites = [kron(I4, projector(t)) for t in 0:3]
const Hint_base = sum(kron(projector(s), projector(s)) for s in 0:3)
const Hhop_a = kron(Gamma, I4)
const Hhop_b = kron(I4, Gamma)

function idx(s, t)
    return s * 4 + t + 1
end

# ===== CONTROL STRUCTURE =====
# Use simple closures that GRAPE can handle natively.
# The alternating structure is built into how we initialize and extract controls.

function make_piecewise_control(values::Vector{Float64}, tlist::Vector{Float64})
    """
    Create a piecewise constant control function.
    """
    function ctrl(t)
        if t <= tlist[1]
            return values[1]
        elseif t >= tlist[end]
            return values[end]
        end
        n_intervals = length(tlist) - 1
        for i in 1:n_intervals
            if tlist[i] <= t < tlist[i+1]
                return values[i]
            end
        end
        return values[end]
    end
    return ctrl
end

function create_controls_and_hamiltonian(n::Int, T::Float64; seed::Int=42)
    """
    Create initial controls and Hamiltonian for GRAPE optimization.

    The Hamiltonian H = sum_i H_i * u_i(t) with 11 controls:
    - Controls 1-4: Va (site potentials for particle a)
    - Controls 5-8: Vb (site potentials for particle b)
    - Control 9: U (interaction)
    - Control 10: Ja (hopping for a)
    - Control 11: Jb (hopping for b)
    """
    Random.seed!(seed)
    dt = T / (2 * n)
    tlist = collect(range(0.0, T, length=2*n+1))

    # Create control value arrays (2n values each for 2n time intervals)
    # Initialize with the alternating structure:
    # - Even intervals (1, 3, 5, ...): Va, Vb, U active
    # - Odd intervals (2, 4, 6, ...): Ja, Jb active

    control_values = Vector{Vector{Float64}}()
    control_ops = Vector{Matrix{ComplexF64}}()

    # Va controls (1-4): active on even intervals
    for s in 1:4
        vals = zeros(2*n)
        for k in 1:n
            vals[2*k - 1] = 0.5 * (rand() - 0.5)  # even interval
        end
        push!(control_values, vals)
        push!(control_ops, P_a_sites[s])
    end

    # Vb controls (5-8): active on even intervals
    for s in 1:4
        vals = zeros(2*n)
        for k in 1:n
            vals[2*k - 1] = 0.5 * (rand() - 0.5)
        end
        push!(control_values, vals)
        push!(control_ops, P_b_sites[s])
    end

    # U control (9): active on even intervals
    vals_U = zeros(2*n)
    for k in 1:n
        vals_U[2*k - 1] = 0.5 * rand() + 0.1
    end
    push!(control_values, vals_U)
    push!(control_ops, Hint_base)

    # Ja control (10): active on odd intervals
    vals_Ja = zeros(2*n)
    for k in 1:n
        vals_Ja[2*k] = 0.5 * rand()
    end
    push!(control_values, vals_Ja)
    push!(control_ops, Hhop_a)

    # Jb control (11): active on odd intervals
    vals_Jb = zeros(2*n)
    for k in 1:n
        vals_Jb[2*k] = 0.5 * rand()
    end
    push!(control_values, vals_Jb)
    push!(control_ops, Hhop_b)

    # Create control functions
    controls = [make_piecewise_control(vals, tlist) for vals in control_values]

    # Build Hamiltonian
    H_drift = zeros(ComplexF64, 16, 16)
    H_terms = Any[H_drift]
    for (op, ctrl) in zip(control_ops, controls)
        push!(H_terms, (op, ctrl))
    end
    H = hamiltonian(H_terms...)

    return H, controls, tlist, dt
end

function extract_controls_from_result(result, n::Int)
    """
    Extract optimized control values from GRAPE result.
    Convert back to the Va, Vb, U, Ja, Jb format used by HEngg_ode.jl.
    """
    opt_ctrls = result.optimized_controls

    Va = zeros(n, 4)
    Vb = zeros(n, 4)
    U = zeros(n)
    Ja = zeros(n)
    Jb = zeros(n)

    # optimized_controls[l] has length 2n+1 (at grid points)
    # We extract values at time intervals (indices 1 to 2n)

    for s in 1:4
        for k in 1:n
            Va[k, s] = opt_ctrls[s][2*k - 1]        # even intervals
            Vb[k, s] = opt_ctrls[4 + s][2*k - 1]    # even intervals
        end
    end

    for k in 1:n
        U[k] = opt_ctrls[9][2*k - 1]   # even intervals
        Ja[k] = opt_ctrls[10][2*k]     # odd intervals
        Jb[k] = opt_ctrls[11][2*k]     # odd intervals
    end

    return Va, Vb, U, Ja, Jb
end

function save_pulse(filename::String, n::Int, T::Float64, Va, Vb, U, Ja, Jb)
    """
    Save pulse in the format expected by HEngg_ode.jl
    """
    dt = T / (2 * n)
    jldsave(filename;
        n = n,
        T = T,
        dt = dt,
        Va = Va,
        Vb = Vb,
        U = U,
        Ja = Ja,
        Jb = Jb
    )
    println("Pulse saved to: $filename")
end

function run_grape_optimization(; n::Int=36, T::Float64=2π, iter_stop::Int=500, seed::Int=42)
    """
    Run GRAPE optimization for the HEngg quantum control problem.
    """
    println("="^60)
    println("GRAPE Optimization for HEngg Quantum Control")
    println("="^60)
    println("\nParameters:")
    println("  n (segments): $n")
    println("  T (total time): $T")
    println("  iter_stop: $iter_stop")
    println("  seed: $seed")

    # Create Hamiltonian and controls
    H, controls, tlist, dt = create_controls_and_hamiltonian(n, T; seed=seed)

    # Initial state: |0,1⟩
    psi0 = zeros(ComplexF64, 16)
    psi0[idx(0, 1)] = 1.0

    # Target state: (|0,1⟩ + |2,3⟩)/√2
    psi_target = zeros(ComplexF64, 16)
    psi_target[idx(0, 1)] = 1.0 / sqrt(2)
    psi_target[idx(2, 3)] = 1.0 / sqrt(2)

    println("\nInitial state: |0,1⟩")
    println("Target state: (|0,1⟩ + |2,3⟩)/√2")
    println("Number of controls: $(length(controls))")
    println("Time steps: $(length(tlist) - 1)")

    # Create trajectory
    traj = Trajectory(psi0, H; target_state=psi_target)

    # Run GRAPE optimization with taylor gradient method
    println("\nStarting GRAPE optimization...")
    println("-"^60)

    result = GRAPE.optimize(
        [traj],
        tlist;
        J_T = J_T_sm,
        iter_stop = iter_stop,
        prop_method = :expprop,
        print_iters = true,
        print_iter_info = ["iter.", "J_T", "ǁ∇Jǁ", "ǁΔϵǁ", "ΔJ", "secs"]
    )

    println("-"^60)
    println("\nOptimization complete!")
    println("  Final J_T: $(result.J_T)")
    println("  Final fidelity: $(1.0 - result.J_T)")
    println("  Converged: $(result.converged)")
    println("  Message: $(result.message)")
    println("  Iterations: $(result.iter)")

    return result, tlist, psi0, psi_target, n, T
end

function plot_results(result, n::Int, T::Float64)
    """
    Plot the optimization results.
    """
    println("\nGenerating plots...")

    # Extract optimized controls
    Va, Vb, U, Ja, Jb = extract_controls_from_result(result, n)

    # Create time points for plotting
    dt = T / (2 * n)
    t_plot = collect(range(0.0, T - dt/2, length=500))

    # Reconstruct control values at each time point
    U_vals = zeros(length(t_plot))
    Ja_vals = zeros(length(t_plot))
    Jb_vals = zeros(length(t_plot))
    V12_vals = zeros(length(t_plot))

    for (i, t) in enumerate(t_plot)
        m = floor(Int, t / dt)
        if m >= 2 * n
            m = 2 * n - 1
        end
        k = m ÷ 2 + 1

        if m % 2 == 0  # even step
            U_vals[i] = U[k]
            Ja_vals[i] = 0.0
            Jb_vals[i] = 0.0
            V12_vals[i] = Va[k, 2] - Va[k, 1]
        else  # odd step
            U_vals[i] = 0.0
            Ja_vals[i] = Ja[k]
            Jb_vals[i] = Jb[k]
            V12_vals[i] = 0.0
        end
    end

    # Plot controls
    p1 = plot(t_plot, U_vals, label="U(t)", color=:blue, linewidth=2,
              xlabel="Time", ylabel="Amplitude", title="Interaction Strength U")

    p2 = plot(t_plot, Ja_vals, label="Ja(t)", color=:red, linewidth=2)
    plot!(t_plot, Jb_vals, label="Jb(t)", color=:orange, linewidth=2,
          xlabel="Time", ylabel="Amplitude", title="Hopping Strengths")

    p3 = plot(t_plot, V12_vals, label="V12(t)", color=:green, linewidth=2,
              xlabel="Time", ylabel="Amplitude", title="Potential Difference V12")

    fidelity_final = 1.0 - result.J_T
    p4 = bar([fidelity_final], label="Final Fidelity", color=:purple,
             xlabel="", ylabel="Fidelity", title="Final Fidelity: $(round(fidelity_final, digits=6))",
             ylim=(0, 1.05))
    hline!([1.0], linestyle=:dash, color=:gray, label="Target")

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
    savefig(fig, "GRAPE_results.png")
    println("Plots saved to: GRAPE_results.png")

    return fig
end

# ===== MAIN EXECUTION =====
if abspath(PROGRAM_FILE) == @__FILE__
    # Run optimization
    result, tlist, psi0, psi_target, n, T = run_grape_optimization(
        n = 36,
        T = 2π,
        iter_stop = 500,
        seed = 42
    )

    # Extract and save optimal pulse
    Va, Vb, U, Ja, Jb = extract_controls_from_result(result, n)
    save_pulse("GRAPE_pulse.jld2", n, T, Va, Vb, U, Ja, Jb)

    # Generate plots
    plot_results(result, n, T)

    println("\n" * "="^60)
    println("COMPLETE")
    println("="^60)
    println("\nOutput files:")
    println("  - GRAPE_pulse.jld2 (optimal pulse, compatible with HEngg_ode.jl)")
    println("  - GRAPE_results.png (visualization)")
end
