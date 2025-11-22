import OrdinaryDiffEq as ODE
import Optimization as OPT
import OptimizationPolyalgorithms as OPA
import SciMLSensitivity as SMS
import Zygote
import Plots

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODE.ODEProblem(lotka_volterra!, u0, tspan, p)
sol = ODE.solve(prob, ODE.Tsit5())

# Plot the solution
Plots.plot(sol)
# Plots.savefig("LV_ode.png")

function loss(p)
    sol = ODE.solve(prob, ODE.Tsit5(), p = p, saveat = tsteps)
    loss = sum(abs2, sol .- 1)
    return loss
end

callback = function (state, l)
    display(l)
    pred = ODE.solve(prob, ODE.Tsit5(), p = state.u, saveat = tsteps)
    plt = Plots.plot(pred, ylim = (0, 6))
    display(plt)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = OPT.OptimizationProblem(optf, p)

result_ode = OPT.solve(optprob, OPA.PolyOpt(),
    callback = callback,
    maxiters = 100)
