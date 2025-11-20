datadir(names...) = joinpath(@__DIR__, names...)

using QuantumControl
using QuantumPropagators: ExpProp

##
ϵ(t) = 0.2 * QuantumControl.Shapes.flattop(t, T=5, t_rise=0.3, func=:blackman);

"""Two-level-system Hamiltonian."""
function tls_hamiltonian(Ω=1.0, ϵ=ϵ)
    σ̂_z = ComplexF64[
        1  0
        0 -1
    ]
    σ̂_x = ComplexF64[
        0  1
        1  0
    ]
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return hamiltonian(Ĥ₀, (Ĥ₁, ϵ))
end;

H = tls_hamiltonian();

tlist = collect(range(0, 5, length=500));

using Plots
Plots.default(
    linewidth               = 3,
    size                    = (550, 300),
    legend                  = :right,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.8)
)

function plot_control(pulse::Vector, tlist)
    plot(tlist, pulse, xlabel="time", ylabel="amplitude", legend=false)
end

plot_control(ϵ::Function, tlist) = plot_control([ϵ(t) for t in tlist], tlist);

fig = plot_control(ϵ, tlist)


##
function ket(label)
    result = Dict("0" => Vector{ComplexF64}([1, 0]), "1" => Vector{ComplexF64}([0, 1]))
    return result[string(label)]
end;

trajectories = [Trajectory(initial_state=ket(0), generator=H, target_state=ket(1))];

using QuantumControl.Functionals: J_T_sm

problem = ControlProblem(
    trajectories=trajectories,
    tlist=tlist,
    iter_stop=500,
    prop_method=ExpProp,
    pulse_options=Dict(),
    J_T=J_T_sm,
    check_convergence=res -> begin
        ((res.J_T < 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
);

guess_dynamics = propagate_trajectory(
    trajectories[1],
    problem.tlist;
    method=ExpProp,
    storage=true,
    observables=(Ψ -> abs.(Ψ) .^ 2,)
)

function plot_population(pop0::Vector, pop1::Vector, tlist)
    fig = plot(tlist, pop0, label="0", xlabel="time", ylabel="population")
    plot!(fig, tlist, pop1; label="1")
end;

fig = plot_population(guess_dynamics[1, :], guess_dynamics[2, :], tlist)


##
using GRAPE

opt_result_GRAPE = @optimize_or_load(
    datadir("TLS", "GRAPE_opt_result.jld2"),
    problem;
    method=GRAPE,
    prop_method=ExpProp,
);

fig = plot_control(opt_result_GRAPE.optimized_controls[1], tlist)  # This is test