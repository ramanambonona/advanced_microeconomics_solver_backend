from sympy import symbols, diff, Eq, Function, Sum, limit, oo, solve, simplify, latex
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict, List, Any, Optional
import numpy as np
from .models import OptimizationResult, SolutionStep

class DynamicSolver:
    def __init__(self):
        self.steps = []

    def solve(self, problem_type: str, T: int, utility_expr: str, constraint_expr: str, 
              parameters: Dict[str, float], beta: float = 0.96) -> Dict[str, Any]:
        
        try:
            # Handle infinite horizon
            if T == float('inf'):
                return self._solve_infinite_horizon(utility_expr, constraint_expr, parameters, beta)
            
            # Generate time-indexed symbols
            c_syms = [symbols(f'c_{t}') for t in range(T)]
            k_syms = [symbols(f'k_{t}') for t in range(T+1)]
            lambda_syms = [symbols(f'lambda_{t}') for t in range(T)]

            # Parse expressions with time substitution
            local_dict = {'beta': beta, **parameters}
            
            # Build Lagrangian
            L = 0
            for t in range(T):
                # Utility at time t
                utility_t = parse_expr(utility_expr, local_dict={**local_dict, 'c': c_syms[t]})
                L += (beta ** t) * utility_t
                
                # Constraint at time t
                constraint_t = parse_expr(constraint_expr, local_dict={
                    **local_dict, 
                    'c': c_syms[t], 
                    'k': k_syms[t], 
                    'k_next': k_syms[t+1]
                })
                L += lambda_syms[t] * constraint_t

            # First Order Conditions
            focs = []
            
            # FOCs for consumption
            for t in range(T):
                foc_c = diff(L, c_syms[t])
                focs.append(Eq(foc_c, 0))
            
            # FOCs for capital (excluding initial capital)
            for t in range(1, T):
                foc_k = diff(L, k_syms[t])
                focs.append(Eq(foc_k, 0))
            
            # FOCs for multipliers (constraints)
            for t in range(T):
                foc_lambda = diff(L, lambda_syms[t])
                focs.append(Eq(foc_lambda, 0))

            # Build Euler equations
            euler_eqs = self._build_euler_equations(utility_expr, constraint_expr, T, parameters, beta)
            
            # Solve system (for small T)
            solution = []
            if T <= 5:  # Limit for symbolic solution
                all_vars = c_syms + k_syms[1:T] + lambda_syms
                solution = solve(focs, all_vars, dict=True)

            # Build steps
            steps = self._build_steps(utility_expr, constraint_expr, focs, euler_eqs, solution, T, beta)

            return {
                "success": True,
                "problem_type": "dynamic",
                "horizon": T,
                "objective": f"∑ₜ₌₀^{T-1} βᵗ U(cₜ)",
                "variables": {
                    "consumption": [str(c) for c in c_syms],
                    "capital": [str(k) for k in k_syms],
                    "multipliers": [str(l) for l in lambda_syms]
                },
                "focs": [str(eq) for eq in focs],
                "euler_equations": [str(eq) for eq in euler_eqs],
                "solution": [{str(k): str(v) for k, v in sol.items()} for sol in solution],
                "steps": [step.dict() for step in steps],
                "steady_state": self._compute_steady_state(utility_expr, constraint_expr, parameters, beta) if T == float('inf') else None
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem_type": "dynamic"
            }

    def _build_euler_equations(self, utility_expr, constraint_expr, T, parameters, beta):
        """Build Euler equations for the dynamic problem"""
        euler_eqs = []
        
        # Symbols for generic periods
        c_t, c_t1 = symbols('c_t c_t1')
        k_t, k_t1, k_t2 = symbols('k_t k_t1 k_t2')
        lambda_t, lambda_t1 = symbols('lambda_t lambda_t1')
        
        local_dict = {'beta': beta, **parameters}
        
        # Utility derivatives
        U_t = parse_expr(utility_expr, local_dict={**local_dict, 'c': c_t})
        U_t1 = parse_expr(utility_expr, local_dict={**local_dict, 'c': c_t1})
        
        dU_dc_t = diff(U_t, c_t)
        dU_dc_t1 = diff(U_t1, c_t1)
        
        # Constraint derivatives
        g_t = parse_expr(constraint_expr, local_dict={**local_dict, 'c': c_t, 'k': k_t, 'k_next': k_t1})
        g_t1 = parse_expr(constraint_expr, local_dict={**local_dict, 'c': c_t1, 'k': k_t1, 'k_next': k_t2})
        
        dg_dc_t = diff(g_t, c_t)
        dg_dk_t1 = diff(g_t, k_t1)
        dg_dc_t1 = diff(g_t1, c_t1)
        dg_dk_t1_constraint = diff(g_t1, k_t1)
        
        # Euler equation
        euler_eq = Eq(
            dU_dc_t + lambda_t * dg_dc_t,
            beta * (dU_dc_t1 + lambda_t1 * dg_dc_t1) + beta * lambda_t1 * dg_dk_t1_constraint
        )
        
        euler_eqs.append(euler_eq)
        return euler_eqs

    def _solve_infinite_horizon(self, utility_expr, constraint_expr, parameters, beta):
        """Solve infinite horizon problem using steady state approach"""
        c, k, k_next = symbols('c k k_next')
        lambda_sym = symbols('lambda')
        
        local_dict = {'beta': beta, **parameters, 'c': c, 'k': k, 'k_next': k_next}
        
        # Current period utility
        U = parse_expr(utility_expr, local_dict=local_dict)
        dU_dc = diff(U, c)
        
        # Constraint
        g = parse_expr(constraint_expr, local_dict=local_dict)
        dg_dc = diff(g, c)
        dg_dk_next = diff(g, k_next)
        
        # Next period (steady state: c = c_next, k = k_next)
        U_next = parse_expr(utility_expr, local_dict={**local_dict, 'c': c, 'k': k_next, 'k_next': k_next})
        dU_dc_next = diff(U_next, c)
        
        g_next = parse_expr(constraint_expr, local_dict={**local_dict, 'c': c, 'k': k_next, 'k_next': k_next})
        dg_dc_next = diff(g_next, c)
        dg_dk_next_constraint = diff(g_next, k_next)
        
        # Euler equation in steady state
        euler_ss = Eq(
            dU_dc + lambda_sym * dg_dc,
            beta * (dU_dc_next + lambda_sym * dg_dc_next) + beta * lambda_sym * dg_dk_next_constraint
        )
        
        # Resource constraint in steady state
        resource_ss = Eq(g, 0)
        
        # Solve for steady state
        steady_state_sol = solve([euler_ss, resource_ss], [c, k], dict=True)
        
        steps = [
            SolutionStep(
                step_name="Problème horizon infini",
                equation="∑ₜ₌₀^∞ βᵗ U(cₜ)",
                explanation="Maximisation de l'utilité escomptée sur horizon infini"
            ),
            SolutionStep(
                step_name="État stationnaire",
                equation="cₜ = cₜ₊₁ = c*, kₜ = kₜ₊₁ = k*",
                explanation="En état stationnaire, les variables sont constantes"
            ),
            SolutionStep(
                step_name="Équation d'Euler en état stationnaire",
                equation=str(euler_ss),
                explanation="Condition d'optimalité intertemporelle"
            )
        ]
        
        return {
            "success": True,
            "problem_type": "dynamic_infinite",
            "steady_state": {str(k): str(v) for k, v in steady_state_sol[0].items()} if steady_state_sol else {},
            "steps": [step.dict() for step in steps],
            "euler_equation": str(euler_ss),
            "resource_constraint": str(resource_ss)
        }

    def _compute_steady_state(self, utility_expr, constraint_expr, parameters, beta):
        """Compute steady state for infinite horizon problems"""
        c, k = symbols('c k')
        local_dict = {'beta': beta, **parameters, 'c': c, 'k': k, 'k_next': k}
        
        # For steady state, we solve the system where variables are constant
        # This is problem-specific and would need to be customized
        return {"steady_state": "À calculer selon les paramètres spécifiques"}

    def _build_steps(self, utility_expr, constraint_expr, focs, euler_eqs, solution, T, beta):
        steps = []
        
        steps.append(SolutionStep(
            step_name="Formulation du problème dynamique",
            equation=f"Max ∑ₜ₌₀^{T-1} ({beta})ᵗ U(cₜ)",
            explanation="Maximisation de l'utilité intertemporelle escomptée"
        ))
        
        steps.append(SolutionStep(
            step_name="Contrainte de ressources",
            equation=f"{constraint_expr} = 0",
            explanation="Contrainte liant consommation et capital entre périodes"
        ))
        
        steps.append(SolutionStep(
            step_name="Construction du Lagrangien",
            equation=f"L = ∑ₜ₌₀^{T-1} [{beta}ᵗ U(cₜ) + λₜ g(cₜ, kₜ, kₜ₊₁)]",
            explanation="Lagrangien avec multiplicateurs pour chaque contrainte"
        ))
        
        for i, foc in enumerate(focs[:3]):  # Show first 3 FOCs
            steps.append(SolutionStep(
                step_name=f"Condition premier ordre {i+1}",
                equation=str(foc),
                explanation="Dérivée partielle du Lagrangien"
            ))
        
        for i, euler_eq in enumerate(euler_eqs):
            steps.append(SolutionStep(
                step_name=f"Équation d'Euler {i+1}",
                equation=str(euler_eq),
                explanation="Condition d'optimalité intertemporelle entre périodes adjacentes"
            ))
        
        if solution:
            steps.append(SolutionStep(
                step_name="Solution du système",
                equation="Système résolu symboliquement",
                explanation="Résolution des conditions premier ordre"
            ))
        
        return steps 
