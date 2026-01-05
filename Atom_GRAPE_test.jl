# qCOMBAT quantum optimal control using GRAPE.jl
# Solves the same problem as HEngg_qCOMBAT.py but using Julia's GRAPE.jl

using LinearAlgebra
using QuantumControl
using QuantumPropagators: ExpProp
using GRAPE
using Plots
using Random

# Define the connectivity matrix (Gamma)
const Γ = Float64[
    0 1 1 0
    1 0 0 1
    1 0 0 1
    0 1 1 0
]


# Helper function for tensor product
⊗(A, B) = kron(A, B)

# Create projector for a specific site
function projector(site_idx::Int)
    e = zeros(ComplexF64, 4, 1)
    e[site_idx + 1] = 1.0  # Julia is 1-indexed
    return e * e'
end

# Projectors for subsystem a (first index)
const P_a_sites = [projector(s) ⊗ I(4) for s in 0:3]

# Projectors for subsystem b (second index)
const P_b_sites = [I(4) ⊗ projector(t) for t in 0:3]

# Interaction Hamiltonian base (diagonal interaction between matching sites)
const H_int_base = sum(projector(s) ⊗ projector(s) for s in 0:3)

# Hopping Hamiltonians
const H_hop_a = Γ ⊗ I(4)
const H_hop_b = I(4) ⊗ Γ

# Define the time-dependent Hamiltonian structure
# In the Python code, the Hamiltonian alternates:
# - Even steps (m=0,2,4,...): Va, Vb, U controls (interaction)
# - Odd steps (m=1,3,5,...): Ja, Jb controls (hopping)

# For GRAPE.jl, we need to define control Hamiltonians
# Let's create piecewise constant controls

function build_hamiltonian(n::Int)
    """
    Build the Hamiltonian with all control terms.
    n: number of pulse segments (each has even+odd step)

    Returns a hamiltonian with drift (zero) and control terms.
    """

    # Drift Hamiltonian (zero in this case)
    H_drift = zeros(ComplexF64, 16, 16)

    # Control Hamiltonians:
    # - 4 controls for Va (on-site potentials for sites 0-3 of subsystem a)
    # - 4 controls for Vb (on-site potentials for sites 0-3 of subsystem b)
    # - 1 control for U (interaction strength)
    # - 1 control for Ja (hopping strength for subsystem a)
    # - 1 control for Jb (hopping strength for subsystem b)

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
    push!(control_ops, H_int_base)

    # Ja control (1)
    push!(control_ops, H_hop_a)

    # Jb control (1)
    push!(control_ops, H_hop_b)

    return control_ops
end

function create_piecewise_controls(n::Int, T::Float64; seed::Int=11)
    """
    Create initial guess for piecewise constant controls.
    n: number of segments
    T: total time

    Returns arrays of control amplitudes and time grid.
    """

    Random.seed!(seed)

    # Time for each half-step
    dt = T / (2 * n)

    # Create time list for 2n steps
    tlist = collect(range(0, T, length=2*n+1))

    # Initialize random controls (matching Python initialization)
    Va = rand(n, 4) .* 2.0 .- 1.0  # uniform(-1, 1)
    Vb = rand(n, 4) .* 2.0 .- 1.0
    U = rand(n) .* 1.8 .+ 0.2       # uniform(0.2, 2.0)
    Ja = rand(n) .* 1.0              # uniform(0, 1)
    Jb = rand(n) .* 1.0

    # Now create piecewise constant controls for each control Hamiltonian
    # The structure is: even steps use Va, Vb, U; odd steps use Ja, Jb

    # For GRAPE.jl, we need one control amplitude per time interval
    # We have 2n intervals total

    controls = []

    # Va controls (4): active on even steps, zero on odd steps
    for s in 1:4
        ctrl = zeros(2*n)
        for k in 1:n
            ctrl[2*k-1] = Va[k, s]  # even step (0-indexed in Python: m=0,2,4,...)
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
        ctrl_Ja[2*k] = Ja[k]  # odd step (0-indexed in Python: m=1,3,5,...)
    end
    push!(controls, ctrl_Ja)

    # Jb control (1): active on odd steps, zero on even steps
    ctrl_Jb = zeros(2*n)
    for k in 1:n
        ctrl_Jb[2*k] = Jb[k]
    end
    push!(controls, ctrl_Jb)

    return controls, tlist
end

function idx(s::Int, t::Int)
    """Convert site indices to state index (0-indexed input)"""
    return s * 4 + t + 1  # +1 for Julia 1-indexing
end

# Define initial and target states
function get_initial_state()
    ψ0 = zeros(ComplexF64, 16)
    ψ0[idx(0, 1)] = 1.0
    return ψ0
end

function get_target_state()
    ψ1 = zeros(ComplexF64, 16)
    ψ1[idx(0, 1)] = 1.0 / sqrt(2)
    ψ1[idx(2, 3)] = 1.0 / sqrt(2)
    return ψ1
end

# Main optimization function
function optimize_pulse(n::Int=36, T::Float64=2π;
                        iter_stop::Int=360,
                        seed::Int=11,
                        λₐ::Float64=1e-5)  # L2 regularization

    println("Setting up optimization problem...")
    println("  Number of segments: $n")
    println("  Total time: $T")
    println("  Iterations: $iter_stop")

    # Create control operators
    control_ops = build_hamiltonian(n)

    # Create initial control guess
    controls, tlist = create_piecewise_controls(n, T; seed=seed)

    # Build Hamiltonian with controls
    # H(t) = Σᵢ uᵢ(t) Hᵢ
    H_drift = zeros(ComplexF64, 16, 16)

    # Create list of (control_op, control_function) tuples
    H_parts = Any[H_drift]
    for (i, op) in enumerate(control_ops)
        # Create piecewise constant control function
        ctrl_vals = controls[i]
        ctrl_func = t -> begin
            # Find which interval we're in
            if t <= 0
                return ctrl_vals[1]
            elseif t >= T
                return ctrl_vals[end]
            else
                # Find the interval index
                interval_idx = min(floor(Int, t / (T / (2*n))) + 1, 2*n)
                return ctrl_vals[interval_idx]
            end
        end
        push!(H_parts, (op, ctrl_func))
    end

    H = hamiltonian(H_parts...)

    # Initial and target states
    ψ_initial = get_initial_state()
    ψ_target = get_target_state()

    # Define trajectory
    trajectory = Trajectory(
        initial_state=ψ_initial,
        generator=H,
        target_state=ψ_target
    )

    # Define the optimization problem
    problem = ControlProblem(
        trajectories=[trajectory],
        tlist=tlist,
        iter_stop=iter_stop,
        prop_method=ExpProp,
        J_T=QuantumControl.Functionals.J_T_sm,  # State-to-state fidelity
        check_convergence=res -> begin
            if res.J_T < 1e-3
                res.converged = true
                res.message = "J_T < 10⁻³"
            end
        end
    )

    println("\nStarting GRAPE optimization...")

    # Run GRAPE optimization
    opt_result = optimize(problem; method=GRAPE)

    println("\nOptimization complete!")
    println("  Final J_T: $(opt_result.J_T)")
    println("  Converged: $(opt_result.converged)")
    println("  Iterations: $(opt_result.iter)")

    return opt_result, problem, tlist
end

# Function to extract and visualize results
function analyze_results(opt_result, problem, tlist)

    println("\n" * "="^60)
    println("RESULTS ANALYSIS")
    println("="^60)

    # Propagate with optimized controls to get fidelity vs time
    trajectory = problem.trajectories[1]

    # Get fidelity at each time point
    states = propagate_trajectory(
        trajectory,
        tlist;
        method=ExpProp,
        storage=true
    )

    ψ_target = trajectory.target_state
    fidelities = [abs2(ψ_target' * states[:, i]) for i in 1:length(tlist)]

    println("\nFidelity progression:")
    println("  Initial: $(fidelities[1])")
    println("  Final:   $(fidelities[end])")

    # Plot fidelity vs time
    p1 = plot(
        tlist,
        fidelities,
        xlabel="Time",
        ylabel="Fidelity",
        title="Fidelity vs Time (GRAPE.jl)",
        legend=false,
        linewidth=2
    )

    # Extract optimized controls
    optimized_controls = opt_result.optimized_controls

    # Plot controls (showing a subset for clarity)
    # Plot U, Ja, Jb controls
    nt = length(tlist) - 1

    # U control (index 9)
    U_ctrl = optimized_controls[9]

    # Ja control (index 10)
    Ja_ctrl = optimized_controls[10]

    # Jb control (index 11)
    Jb_ctrl = optimized_controls[11]

    # Ensure arrays match in length
    n_ctrl = min(nt, length(U_ctrl), length(Ja_ctrl), length(Jb_ctrl))
    t_plot = tlist[1:n_ctrl]

    p2 = plot(
        t_plot,
        U_ctrl[1:n_ctrl],
        xlabel="Time",
        ylabel="Amplitude",
        title="Optimized Controls",
        label="U (interaction)",
        linewidth=2,
        seriestype=:steppost
    )
    plot!(p2, t_plot, Ja_ctrl[1:n_ctrl], label="Jₐ (hopping a)", linewidth=2, seriestype=:steppost)
    plot!(p2, t_plot, Jb_ctrl[1:n_ctrl], label="Jᵦ (hopping b)", linewidth=2, seriestype=:steppost)

    # Combine plots
    p = plot(p1, p2, layout=(2, 1), size=(800, 600))

    display(p)

    return fidelities, optimized_controls, p
end

# Run the optimization
println("="^60)
println("HEngg Quantum Control Problem - GRAPE.jl Implementation")
println("="^60)

opt_result, problem, tlist = optimize_pulse(36, 2π; iter_stop=360, seed=11)

# Analyze and visualize results
fidelities, optimized_controls, fig = analyze_results(opt_result, problem, tlist)

# Save the figure
# savefig(fig, "HEngg_GRAPE_results.png")
# println("\nFigure saved as 'HEngg_GRAPE_results.png'")
