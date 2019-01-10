using JuMP, AmplNLWriter, MathOptInterfaceToMathProgBase, Test, Ipopt
const MOI = JuMP.MOI

# model = JuMP.Model(
#     with_optimizer(
#         MathOptInterfaceToMathProgBase.Optimizer,
#         solver = AmplNLWriter.AmplNLSolver(Ipopt.amplexe)
#     )
# )

model = JuMP.Model(with_optimizer(MathOptInterfaceToMathProgBase.Optimizer, solver = nothing))
MOI.set(model, MathOptInterfaceToMathProgBase.MPBSolver(),
    AmplNLWriter.AmplNLSolver(Ipopt.amplexe)
)

N = 1000
h = 1/N
alpha = 350

@variables(model, begin
    -1 <= t[1:(N + 1)] <= 1
    -0.05 <= x[1:(N + 1)] <= 0.05
    u[1:(N + 1)]
end)
@NLobjective(model, Min, sum(0.5 * h * (u[i + 1]^2 + u[i]^2) +
    0.5 * alpha * h * (cos(t[i + 1]) + cos(t[i])) for i in 1:N)
)
@NLconstraint(model, [i = 1:N],
    x[i + 1] - x[i] - 0.5 * h * (sin(t[i + 1]) + sin(t[i])) == 0
)
@constraint(model, [i = 1:N],
    t[i + 1] - t[i] - 0.5 * h * u[i + 1] - 0.5 * h * u[i] == 0
)
JuMP.optimize!(model)

# @test JuMP.termination_status(model) == MOI.OPTIMAL
# @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
@test JuMP.objective_value(model) ≈ 350.0
