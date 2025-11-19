using DifferentialEquations, Plots, DiffEqParamEstim, Optimization, OptimizationMOI,
      OptimizationNLopt, NLopt

function f(du, u, p, t)
    du[1] = p[1] * u[1] - u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5]
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5())

t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

obj = build_loss_objective(prob, Tsit5(), L2Loss(t, data), Optimization.AutoForwardDiff())


##
opt = Opt(:LN_COBYLA, 1)
optprob = Optimization.OptimizationProblem(obj, [1.3])
res = solve(optprob, opt)


##
opt = Opt(:GN_ESCH, 1)
lower_bounds!(opt, [0.0])
upper_bounds!(opt, [5.0])
xtol_rel!(opt, 1e-3)
maxeval!(opt, 100000)
res = solve(optprob, opt)


##
optprob = Optimization.OptimizationProblem(obj, [0.2], lb = [-1.0], ub = [5.0])
res = solve(optprob,
            OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
                                                        "algorithm" => :GN_ISRES,
                                                        "xtol_rel" => 1e-3,
                                                        "maxeval" => 10000))


##
opt = Opt(:LN_BOBYQA, 1)
min_objective!(opt, obj)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
