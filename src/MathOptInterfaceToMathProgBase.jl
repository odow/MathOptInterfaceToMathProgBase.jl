module MathOptInterfaceToMathProgBase

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathProgBase
const MPB = MathProgBase.SolverInterface

MOIU.@model(InnerModel,
    (MOI.ZeroOne, MOI.Integer),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (),
    (),
    (MOI.SingleVariable,),
    (),
    (),
    ()
)

"Attribute for the MathProgBase solver."
struct MPBSolver <: MOI.AbstractModelAttribute end

"""
    Optimizer(; solver)

# Example

    model = Optimizer(solver = AmplNLWriter("/usr/bin/bonmin"))
"""
function Optimizer(; solver)
    model = MOIU.UniversalFallback(InnerModel{Float64}())
    return model
end

# Make `Model` a constant for use in the rest of the file.
const Model = MOIU.UniversalFallback{InnerModel{Float64}}
Base.show(io::IO, ::Model) = println(io, "A MathProgBase model")

set_to_bounds(set::MOI.LessThan) = (-Inf, set.upper)
set_to_bounds(set::MOI.GreaterThan) = (set.lower, Inf)
set_to_bounds(set::MOI.EqualTo) = (set.value, set.value)
set_to_bounds(set::MOI.Interval) = (set.lower, set.upper)
set_to_cat(set::MOI.ZeroOne) = :Bin
set_to_cat(set::MOI.Integer) = :Int

struct NLPEvaluator{T <: MOI.AbstractNLPEvaluator} <: MPB.AbstractNLPEvaluator
    inner::T
    variable_map::Dict{MOI.VariableIndex, Int}
end

"""
MathProgBase expects expressions with variables denoted by `x[i]` for contiguous
`i`. However, JuMP 0.19 creates expressions with `x[MOI.VariableIndex(i)]`. So
we have to recursively walk the expression replacing instances of
MOI.VariableIndex by a corresponding integer.
"""
function replace_variableindex_by_int(variable_map, expr::Expr)
    for (i, arg) in enumerate(expr.args)
        expr.args[i] = replace_variableindex_by_int(variable_map, arg)
    end
    return expr
end
function replace_variableindex_by_int(variable_map, expr::MOI.VariableIndex)
    return variable_map[expr]
end
replace_variableindex_by_int(variable_map, expr) = expr

function MPB.initialize(d::NLPEvaluator, requested_features::Vector{Symbol})
    MOI.initialize(d.inner, requested_features)
    return
end

function MPB.features_available(d::NLPEvaluator)
    return MOI.features_available(d.inner)
end

function MPB.eval_f(d::NLPEvaluator, x)
    return MOI.eval_objective(d.inner, x)
end

function MPB.eval_g(d::NLPEvaluator, g, x)
    return MOI.eval_constraint(d.inner, g, x)
end

function MPB.eval_grad_f(d::NLPEvaluator, g, x)
    return MOI.eval_objective_gradient(d.inner, g, x)
end

function MPB.jac_structure(d::NLPEvaluator)
    return MOI.jacobian_structure(d.inner)
end

function MPB.obj_expr(d::NLPEvaluator)
    expr = MOI.objective_expr(d.inner)
    return replace_variableindex_by_int(d.variable_map, expr)
end

function MPB.constr_expr(d::NLPEvaluator, i)
    expr = MOI.constraint_expr(d.inner, i)
    return replace_variableindex_by_int(d.variable_map, expr)
end

function MOI.optimize!(model::Model)
    mpb_solver = MOI.get(model, MPBSolver())

    # Get the variable data.
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    num_var = length(variables)
    x_l = fill(-Inf, num_var)
    x_u = fill(Inf, num_var)
    x_cat = fill(:Cont, num_var)

    variable_map = Dict{MOI.VariableIndex, Int}()
    for (i, variable) in enumerate(variables)
        variable_map[variable] = i
    end

    for set_type in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64})
        for c_ref in MOI.get(model,
            MOI.ListOfConstraintIndices{MOI.SingleVariable, set_type}())
            c_func = MOI.get(model, MOI.ConstraintFunction(), c_ref)
            c_set = MOI.get(model, MOI.ConstraintSet(), c_ref)
            v_index = variable_map[c_func.variable]
            lower, upper = set_to_bounds(c_set)
            x_l[v_index] = lower
            x_u[v_index] = upper
        end
    end

    for set_type in (MOI.ZeroOne, MOI.Integer)
        for c_ref in MOI.get(model,
            MOI.ListOfConstraintIndices{MOI.SingleVariable, set_type}())
            c_func = MOI.get(model, MOI.ConstraintFunction(), c_ref)
            c_set = MOI.get(model, MOI.ConstraintSet(), c_ref)
            v_index = variable_map[c_func.variable]
            x_cat[v_index] = set_to_cat(c_set)
        end
    end

    # Get the optimzation sense.
    opt_sense = MOI.get(model, MOI.ObjectiveSense())
    sense = opt_sense == MOI.MAX_SENSE ? :Max : :Min

    nlp_block = try
        MOI.get(model, MOI.NLPBlock())
    catch ex
        error("Expected a NLPBLock.")
    end

    # Extract constraint bounds.
    num_con = length(nlp_block.constraint_bounds)
    g_l = fill(-Inf, num_con)
    g_u = fill(Inf, num_con)
    for (i, bound) in enumerate(nlp_block.constraint_bounds)
        g_l[i] = bound.lower
        g_u[i] = bound.upper
    end

    mpb_model = MPB.NonlinearModel(mpb_solver)
    MPB.loadproblem!(
        mpb_model, num_var, num_con, x_l, x_u, g_l, g_u, sense,
        NLPEvaluator(nlp_block.evaluator, variable_map)
    )

    MPB.setvartype!(mpb_model, x_cat)

    MPB.optimize!(mpb_model)

    # Handle statuses.
    status = MPB.status(mpb_model)
    termination_status = MOI.OTHER_ERROR
    primal_status = MOI.NO_SOLUTION
    dual_status = MOI.NO_SOLUTION
    if status == :Optimal
        termination_status = MOI.OPTIMAL
        primal_status = MOI.FEASIBLE_POINT
    elseif status == :Infeasible
        termination_status = MOI.INFEASIBLE
    elseif status == :Unbounded
        termination_status = MOI.DUAL_INFEASIBLE
    elseif status == :UserLimit
        termination_status = MOI.OTHER_LIMIT
    elseif status == :Error
        termination_status = MOI.OTHER_ERROR
    end

    # MOI.set(model, MOI.TerminationStatus(), termination_status)
    # MOI.set(model, MOI.PrimalStatus(), primal_status)
    # MOI.set(model, MOI.DualStatus(), dual_status)

    x_solution = MPB.getsolution(mpb_model)
    for (variable, sol) in zip(variables, x_solution)
        # MOI.set(model, MOI.VariablePrimal(), variable, sol)
    end

    obj_value = MPB.getobjval(model)
    # MOI.set(model, MOI.ObjectiveValue(), obj_value)

    return
end

end
