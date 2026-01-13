# HEngg_GRAPE.jl
# Implements the same quantum control problem as HEngg_ode.jl
# but using GRAPE.jl package instead of custom GRAPE implementation

using LinearAlgebra
using QuantumControl
using QuantumPropagators: ExpProp
using GRAPE
using Plots
using Random

# ===== QUANTUM SYSTEM SETUP =====
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

# ===== BUILD CONTROL HAMILTONIAN STRUCTURE =====
function build_control_operators()
    """
    Build the control operators for GRAPE.jl
    Returns a list of control operators in order:
    - 4 operators for Va (on-site potentials for subsystem a, sites 0-3)
    - 4 operators for Vb (on-site potentials for subsystem b, sites 0-3)
    - 1 operator for U (interaction strength)
    - 1 operator for Ja (hopping strength for subsystem a)
    - 1 operator for Jb (hopping strength for subsystem b)
    Total: 11 control operators
    """
    control_ops = []

    # Va controls (4)
    for s in 1:4
        push!(control_ops, P_a_sites[s])
    end

    # Vb controls (4)
    for t in 1:4
        push!(control_ops, P_b_sites[t])
    end

    # U control (1)
    push!(control_ops, Hint_base)

    # Ja control (1)
    push!(control_ops, Hhop_a)

    # Jb control (1)
    push!(control_ops, Hhop_b)

    return control_ops
end

function create_initial_controls(n::Int, T::Float64; use_optimized::Bool=false, seed::Int=11)
    """
    Create initial control guess
    n: number of segments (each segment has even + odd step)
    T: total time
    use_optimized: if true, use the optimized values from HEngg_ode.jl as initial guess

    Returns arrays of control amplitudes for 2n time steps
    """

    dt = T / (2 * n)
    
    # Random initialization
    Random.seed!(seed)
    Va = rand(n, 4) .* 2.0 .- 1.0
    Vb = rand(n, 4) .* 2.0 .- 1.0
    U = rand(n) .* 1.8 .+ 0.2
    Ja = rand(n) .* 1.0
    Jb = rand(n) .* 1.0

    # Create piecewise constant controls for GRAPE.jl
    # Structure: even steps (0,2,4,...) use Va, Vb, U; odd steps (1,3,5,...) use Ja, Jb
    controls = []

    # Va controls (4): active on even steps, zero on odd steps
    for s in 1:4
        ctrl = zeros(2*n)
        for k in 1:n
            ctrl[2*k-1] = Va[k, s]  # even step: indices 1,3,5,...
        end
        push!(controls, ctrl)
    end

    # Vb controls (4): active on even steps, zero on odd steps
    for t in 1:4
        ctrl = zeros(2*n)
        for k in 1:n
            ctrl[2*k-1] = Vb[k, t]
        end
        push!(controls, ctrl)
    end

    # U control (1): active on even steps, zero on odd steps
    ctrl_U = zeros(2*n)
    for k in 1:n
        ctrl_U[2*k-1] = U[k]
    end
    push!(controls, ctrl_U)

    # Ja control (1): active on odd steps, zero on even steps
    ctrl_Ja = zeros(2*n)
    for k in 1:n
        ctrl_Ja[2*k] = Ja[k]  # odd step: indices 2,4,6,...
    end
    push!(controls, ctrl_Ja)

    # Jb control (1): active on odd steps, zero on even steps
    ctrl_Jb = zeros(2*n)
    for k in 1:n
        ctrl_Jb[2*k] = Jb[k]
    end
    push!(controls, ctrl_Jb)

    return controls
end

function build_hamiltonian(controls, control_ops, T::Float64, n::Int)
    """
    Build time-dependent Hamiltonian for GRAPE.jl
    """
    # Drift Hamiltonian (zero in this case)
    H_drift = zeros(ComplexF64, 16, 16)

    # Create Hamiltonian with piecewise constant controls
    H_parts = Any[H_drift]

    for (i, op) in enumerate(control_ops)
        ctrl_vals = controls[i]

        # Create piecewise constant control function
        ctrl_func = t -> begin
            if t <= 0
                return ctrl_vals[1]
            elseif t >= T
                return ctrl_vals[end]
            else
                # Find the interval index (2n intervals)
                interval_idx = min(floor(Int, t / (T / (2*n))) + 1, 2*n)
                return ctrl_vals[interval_idx]
            end
        end

        push!(H_parts, (op, ctrl_func))
    end

    return hamiltonian(H_parts...)
end

# ===== OPTIMIZATION =====
function optimize_with_grape(n::Int=36, T::Float64=2π;
                             iter_stop::Int=360,
                             use_optimized_init::Bool=false,
                             seed::Int=11)

    println("="^60)
    println("GRAPE.jl OPTIMIZATION")
    println("="^60)
    println("\nProblem setup:")
    println("  Number of segments: $n")
    println("  Total time: $T")
    println("  Iterations: $iter_stop")
    println("  Using optimized init: $use_optimized_init")

    # Create control operators
    control_ops = build_control_operators()
    println("  Number of controls: $(length(control_ops))")

    # Create initial controls
    controls = create_initial_controls(n, T; use_optimized=use_optimized_init, seed=seed)

    # Create time grid
    tlist = collect(range(0, T, length=2*n+1))

    # Build Hamiltonian
    H = build_hamiltonian(controls, control_ops, T, n)

    # Initial and target states
    psi0 = zeros(ComplexF64, 16)
    psi0[idx(0, 1)] = 1.0

    psi_target = zeros(ComplexF64, 16)
    psi_target[idx(0, 1)] = 1.0 / sqrt(2)
    psi_target[idx(2, 3)] = 1.0 / sqrt(2)

    println("\nInitial state: |0,1⟩")
    println("Target state: (|0,1⟩ + |2,3⟩)/√2")

    # Define trajectory
    trajectory = Trajectory(
        initial_state=psi0,
        generator=H,
        target_state=psi_target
    )

    # Define optimization problem
    problem = ControlProblem(
        trajectories=[trajectory],
        tlist=tlist,
        iter_stop=iter_stop,
        prop_method=ExpProp,
        J_T=QuantumControl.Functionals.J_T_sm,  # State-to-state fidelity
        check_convergence=res -> begin
            if res.J_T < 1e-6
                res.converged = true
                res.message = "J_T < 10⁻⁶"
            end
            # Print progress every 50 iterations
            if res.iter % 50 == 0 || res.iter <= 5 || res.iter > iter_stop - 5
                fidelity = 1.0 - res.J_T
                println("  Iteration $(res.iter): Fidelity = $(round(fidelity, digits=10))")
            end
        end
    )

    println("\nStarting GRAPE optimization...")

    # Run GRAPE optimization
    opt_result = optimize(problem; method=GRAPE)

    println("\nOptimization complete!")
    fidelity_final = 1.0 - opt_result.J_T
    println("  Final fidelity: $fidelity_final")
    println("  Converged: $(opt_result.converged)")
    println("  Iterations: $(opt_result.iter)")

    return opt_result, problem, tlist, psi0, psi_target
end

function analyze_results(opt_result, problem, tlist, psi0, psi_target)
    """
    Extract and analyze the optimization results
    """
    println("\n" * "="^60)
    println("RESULTS ANALYSIS")
    println("="^60)

    # Get optimized controls and propagate
    trajectory = problem.trajectories[1]

    # Propagate with optimized controls
    states = propagate_trajectory(
        trajectory,
        tlist;
        method=ExpProp,
        storage=true
    )

    # Calculate fidelities at each time
    fidelities = [abs2(psi_target' * states[:, i]) for i in 1:length(tlist)]

    println("\nFidelity progression:")
    println("  Initial: $(round(fidelities[1], digits=10))")
    println("  Final:   $(round(fidelities[end], digits=10))")

    # Extract optimized controls
    optimized_controls = opt_result.optimized_controls

    return fidelities, optimized_controls, states
end

function plot_results(tlist, fidelities, optimized_controls)
    """
    Plot fidelity and control amplitudes
    """
    println("\nGenerating plots...")

    # Plot 1: Fidelity vs time
    p1 = plot(
        tlist,
        fidelities,
        xlabel="Time",
        ylabel="Fidelity",
        title="Fidelity vs Time (GRAPE.jl)",
        legend=false,
        linewidth=2,
        color=:blue
    )

    # Plot 2: Selected controls (U, Ja, Jb)
    nt = length(tlist) - 1
    t_plot = tlist[1:nt]

    # U control (index 9)
    U_ctrl = optimized_controls[9][1:nt]

    # Ja control (index 10)
    Ja_ctrl = optimized_controls[10][1:nt]

    # Jb control (index 11)
    Jb_ctrl = optimized_controls[11][1:nt]

    p2 = plot(
        t_plot,
        U_ctrl,
        xlabel="Time",
        ylabel="Amplitude",
        title="Optimized Controls",
        label="U (interaction)",
        linewidth=2,
        seriestype=:steppost,
        color=:red
    )
    plot!(p2, t_plot, Ja_ctrl, label="Jₐ (hopping a)", linewidth=2, seriestype=:steppost, color=:green)
    plot!(p2, t_plot, Jb_ctrl, label="Jᵦ (hopping b)", linewidth=2, seriestype=:steppost, color=:orange)

    # Combine plots
    fig = plot(p1, p2, layout=(2, 1), size=(800, 800))

    savefig(fig, "HEngg_GRAPE_results.png")
    println("Plot saved: HEngg_GRAPE_results.png")

    return fig
end

# ===== MAIN EXECUTION =====
println("="^60)
println("HEngg Quantum Control - GRAPE.jl Implementation")
println("="^60)

# Run optimization starting from the optimized pulse in HEngg_ode.jl
println("\nMode: Starting from optimized pulse (for comparison)")
opt_result, problem, tlist, psi0, psi_target = optimize_with_grape(
    36, 2π;
    iter_stop=360,
    use_optimized_init=true,
    seed=11
)

# Analyze results
fidelities, optimized_controls, states = analyze_results(opt_result, problem, tlist, psi0, psi_target)

# Plot results
fig = plot_results(tlist, fidelities, optimized_controls)

println("\n" * "="^60)
println("COMPLETE")
println("="^60)
