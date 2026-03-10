from sympy import symbols, diff, Eq, Sum, Matrix, simplify, latex
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict, List, Any, Optional
import numpy as np
from .models import OptimizationResult, SolutionStep

class StochasticSolver:
    def __init__(self):
        self.steps = []

    def solve(self, problem_type: str, T: int, S: int, P: List[List[float]], 
              utility_type: str, constraints: List[str], 
              parameters: Dict[str, float]) -> Dict[str, Any]:
        
        try:
            if problem_type == "markov":
                return self._solve_markov_problem(T, S, P, utility_type, constraints, parameters)
            elif problem_type == "vector":
                return self._solve_vector_problem(T, S, P, utility_type, constraints, parameters)
            else:
                return {"success": False, "error": f"Type de problème non supporté: {problem_type}"}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem_type": "stochastic"
            }

    def _solve_markov_problem(self, T: int, S: int, P: List[List[float]], 
                             utility_type: str, constraints: List[str], 
                             parameters: Dict[str, float]) -> Dict[str, Any]:
        """Solve Markov decision process problem"""
        
        # Generate state-contingent symbols
        c_syms = [[symbols(f'c_{t}_s{s}') for s in range(S)] for t in range(T)]
        k_syms = [[symbols(f'k_{t}_s{s}') for s in range(S)] for t in range(T+1)]
        lambda_syms = [[symbols(f'lambda_{t}_s{s}') for s in range(S)] for t in range(T)]
        z_syms = [symbols(f'z_s{s}') for s in range(S)]

        beta = parameters.get('beta', 0.96)
        
        # Build expected utility Lagrangian
        L = 0
        for t in range(T):
            for i in range(S):  # current state
                # Current utility
                if utility_type == "crra":
                    gamma = parameters.get('gamma', 2.0)
                    U_t = c_syms[t][i]**(1-gamma) / (1-gamma)
                elif utility_type == "log":
                    U_t = sympy.log(c_syms[t][i])
                else:  # custom
                    U_t = parse_expr(utility_type, {
                        'c': c_syms[t][i],
                        'k': k_syms[t][i],
                        'z': z_syms[i],
                        **parameters
                    })
                
                # Expected continuation value
                exp_future = 0
                for j in range(S):  # next state
                    if utility_type == "crra":
                        U_t1 = c_syms[t+1][j]**(1-gamma) / (1-gamma) if t < T-1 else 0
                    elif utility_type == "log":
                        U_t1 = sympy.log(c_syms[t+1][j]) if t < T-1 else 0
                    else:
                        U_t1 = parse_expr(utility_type, {
                            'c': c_syms[t+1][j],
                            'k': k_syms[t+1][j],
                            'z': z_syms[j],
                            **parameters
                        }) if t < T-1 else 0
                    
                    exp_future += P[i][j] * U_t1
                
                # Add to Lagrangian
                L += (beta ** t) * (U_t + beta * exp_future)
                
                # Add constraints with multipliers
                for constraint in constraints:
                    g = parse_expr(constraint, {
                        'c': c_syms[t][i],
                        'k': k_syms[t][i],
                        'k_next': k_syms[t+1][j] if t < T-1 else symbols(f'k_{T}_s{j}'),
                        'z': z_syms[i],
                        'z_next': z_syms[j],
                        **parameters
                    })
                    L += lambda_syms[t][i] * g

        # Build FOCs
        focs = []
        euler_eqs = []

        # FOCs for consumption
        for t in range(T):
            for i in range(S):
                foc_c = diff(L, c_syms[t][i])
                focs.append(Eq(foc_c, 0))

        # FOCs for capital
        for t in range(1, T):
            for i in range(S):
                foc_k = diff(L, k_syms[t][i])
                focs.append(Eq(foc_k, 0))

        # Build stochastic Euler equations
        for t in range(T-1):
            for i in range(S):
                # Current marginal utility
                if utility_type == "crra":
                    mu_t = c_syms[t][i]**(-parameters.get('gamma', 2.0))
                elif utility_type == "log":
                    mu_t = 1/c_syms[t][i]
                else:
                    U_t = parse_expr(utility_type, {'c': c_syms[t][i], **parameters})
                    mu_t = diff(U_t, c_syms[t][i])
                
                # Expected future marginal utility
                exp_future_mu = 0
                for j in range(S):
                    if utility_type == "crra":
                        mu_t1 = c_syms[t+1][j]**(-parameters.get('gamma', 2.0))
                    elif utility_type == "log":
                        mu_t1 = 1/c_syms[t+1][j]
                    else:
                        U_t1 = parse_expr(utility_type, {'c': c_syms[t+1][j], **parameters})
                        mu_t1 = diff(U_t1, c_syms[t+1][j])
                    
                    # Return on capital
                    R_expr = self._parse_return(constraints[0], parameters, k_syms[t+1][j], z_syms[j])
                    exp_future_mu += P[i][j] * mu_t1 * R_expr
                
                euler_eq = Eq(mu_t, beta * exp_future_mu)
                euler_eqs.append(euler_eq)

        # Build steps
        steps = self._build_stochastic_steps(T, S, P, utility_type, constraints, focs, euler_eqs)

        return {
            "success": True,
            "problem_type": "markov",
            "horizon": T,
            "states": S,
            "transition_matrix": P,
            "variables": {
                "consumption": [[str(c) for c in row] for row in c_syms],
                "capital": [[str(k) for k in row] for row in k_syms],
                "shocks": [str(z) for z in z_syms]
            },
            "focs": [str(eq) for eq in focs[:10]],  # Limit output
            "euler_equations": [str(eq) for eq in euler_eqs],
            "steps": [step.dict() for step in steps],
            "stationary_distribution": self._compute_stationary_distribution(P)
        }

    def _solve_vector_problem(self, T: int, S: int, P: List[List[float]], 
                             utility_type: str, constraints: List[str], 
                             parameters: Dict[str, float]) -> Dict[str, Any]:
        """Solve vectorized stochastic problem"""
        
        # This would implement the vectorstoch.py functionality
        # For now, return a simplified version
        steps = [
            SolutionStep(
                step_name="Problème stochastique vectoriel",
                equation=f"Max E[∑ₜ₌₀^T βᵗ U(cₜ)] avec {S} états",
                explanation="Maximisation de l'utilité espérée sous incertitude Markovienne"
            ),
            SolutionStep(
                step_name="Matrice de transition",
                equation=f"P = {P}",
                explanation="Probabilités de transition entre états"
            ),
            SolutionStep(
                step_name="Équations d'Euler stochastiques",
                equation="U'(cₜ) = β E[U'(cₜ₊₁) Rₜ₊₁ | Iₜ]",
                explanation="Condition d'optimalité intertemporelle sous incertitude"
            )
        ]

        return {
            "success": True,
            "problem_type": "vector",
            "horizon": T,
            "states": S,
            "transition_matrix": P,
            "steps": [step.dict() for step in steps],
            "method": "Méthode vectorielle - implémentation en cours"
        }

    def _parse_return(self, constraint: str, parameters: Dict, k_next, z_next):
        """Parse return on capital from constraint"""
        # Simple parser for common production functions
        if 'A*k**alpha' in constraint or 'A*z*k**alpha' in constraint:
            alpha = parameters.get('alpha', 0.33)
            delta = parameters.get('delta', 0.08)
            A = parameters.get('A', 1.0)
            return A * z_next * alpha * k_next**(alpha-1) + (1 - delta)
        else:
            # Default return (simplified)
            return symbols('R')

    def _compute_stationary_distribution(self, P: List[List[float]]) -> List[float]:
        """Compute stationary distribution of Markov chain"""
        try:
            P_array = np.array(P)
            # Find eigenvector for eigenvalue 1
            eigenvalues, eigenvectors = np.linalg.eig(P_array.T)
            stationary = eigenvectors[:, np.isclose(eigenvalues, 1)]
            stationary = stationary[:, 0] / stationary[:, 0].sum()
            return stationary.real.tolist()
        except:
            return [1/len(P)] * len(P)  # Uniform distribution as fallback

    def _build_stochastic_steps(self, T, S, P, utility_type, constraints, focs, euler_eqs):
        steps = []
        
        steps.append(SolutionStep(
            step_name="Problème stochastique",
            equation=f"Max E[∑ₜ₌₀^{T-1} βᵗ U(cₜ)]",
            explanation="Maximisation de l'utilité espérée sous incertitude"
        ))
        
        steps.append(SolutionStep(
            step_name="Processus Markovien",
            equation=f"S = {S} états, matrice de transition P",
            explanation="Incertitude modélisée par une chaîne de Markov"
        ))
        
        steps.append(SolutionStep(
            step_name="Utilité espérée",
            equation="E[U] = ∑ᵢ πᵢ U(cᵢ) + β ∑ᵢ∑ⱼ Pᵢⱼ U(cⱼ)",
            explanation="Formulation récursive de l'utilité espérée"
        ))
        
        steps.append(SolutionStep(
            step_name="Équation d'Euler stochastique",
            equation="U'(cₜ) = β E[U'(cₜ₊₁) Rₜ₊₁ | Iₜ]",
            explanation="Condition d'optimalité avec espérance conditionnelle"
        ))
        
        for i, euler_eq in enumerate(euler_eqs[:3]):  # Show first 3
            steps.append(SolutionStep(
                step_name=f"Équation d'Euler {i+1}",
                equation=str(euler_eq),
                explanation="Forme spécifique pour le problème courant"
            ))
        
        return steps

    def monte_carlo_simulation(self, T: int, S: int, P: List[List[float]], 
                              policy_function: str, parameters: Dict[str, float], 
                              n_simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo simulation of stochastic model"""
        
        np.random.seed(42)  # For reproducibility
        
        # Simulate state paths
        state_paths = []
        for _ in range(n_simulations):
            path = [np.random.choice(S)]  # Initial state
            for t in range(1, T):
                current_state = path[-1]
                next_state = np.random.choice(S, p=P[current_state])
                path.append(next_state)
            state_paths.append(path)
        
        # Compute statistics
        state_paths = np.array(state_paths)
        mean_path = state_paths.mean(axis=0)
        std_path = state_paths.std(axis=0)
        
        return {
            "success": True,
            "method": "monte_carlo",
            "n_simulations": n_simulations,
            "mean_path": mean_path.tolist(),
            "std_path": std_path.tolist(),
            "state_paths_sample": state_paths[:10].tolist()  # Sample of paths
        } 
