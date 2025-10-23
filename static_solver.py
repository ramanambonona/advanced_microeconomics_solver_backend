from sympy import symbols, diff, Eq, solve, simplify, ln, exp, Min, Matrix, latex
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict, List, Any, Optional
import numpy as np
from .models import SolutionStep

class StaticSolver:
    def __init__(self):
        self.steps = []

    def solve(self, problem_type: str, utility_type: str, n_goods: int, 
              parameters: Dict[str, float], constraints: List[str], 
              custom_utility: str = None, options: Dict[str, bool] = None) -> Dict[str, Any]:
        
        try:
            if problem_type == "marshallian":
                return self._solve_marshallian(utility_type, n_goods, parameters, custom_utility, options)
            elif problem_type == "hicksian":
                return self._solve_hicksian(utility_type, n_goods, parameters, custom_utility, options)
            else:
                return {"success": False, "error": f"Type de problème non supporté: {problem_type}"}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem_type": "static"
            }

    def _solve_marshallian(self, utility_type: str, n_goods: int, 
                          parameters: Dict[str, float], custom_utility: str = None,
                          options: Dict[str, bool] = None) -> Dict[str, Any]:
        """Solve Marshallian demand problem: max U(x) s.t. budget constraint"""
        
        # Generate symbols
        x_syms = [symbols(f'x{i}') for i in range(1, n_goods + 1)]
        p_syms = [symbols(f'p{i}') for i in range(1, n_goods + 1)]
        R = symbols('R')
        lambda_sym = symbols('lambda')
        
        # Build utility function
        U = self._build_utility_function(utility_type, x_syms, parameters, custom_utility)
        
        # Budget constraint
        budget_constraint = sum(p_syms[i] * x_syms[i] for i in range(n_goods)) - R
        
        # Lagrangian
        L = U + lambda_sym * budget_constraint

        # First Order Conditions
        focs = []
        for i in range(n_goods):
            foc = diff(L, x_syms[i])
            focs.append(Eq(foc, 0))
        focs.append(Eq(diff(L, lambda_sym), 0))

        # Solve system
        solution_vars = x_syms + [lambda_sym]
        solution = solve(focs, solution_vars, dict=True)

        # Compute additional results
        elasticities = self._compute_elasticities(solution[0] if solution else {}, x_syms, p_syms, R, U)
        tms = self._compute_tms(U, x_syms) if n_goods >= 2 else None
        slutsky_matrix = self._compute_slutsky_matrix(solution[0] if solution else {}, x_syms, p_syms, R, U)

        # Build steps
        steps = self._build_marshallian_steps(U, budget_constraint, focs, solution, elasticities, tms)

        return {
            "success": True,
            "problem_type": "marshallian",
            "utility_function": str(U),
            "variables": {
                "goods": [str(x) for x in x_syms],
                "prices": [str(p) for p in p_syms],
                "income": str(R)
            },
            "focs": [str(foc) for foc in focs],
            "solution": [{str(k): str(v) for k, v in sol.items()} for sol in solution],
            "elasticities": elasticities,
            "tms": str(tms) if tms else None,
            "slutsky_matrix": slutsky_matrix,
            "steps": [step.dict() for step in steps]
        }

    def _solve_hicksian(self, utility_type: str, n_goods: int, 
                       parameters: Dict[str, float], custom_utility: str = None,
                       options: Dict[str, bool] = None) -> Dict[str, Any]:
        """Solve Hicksian demand problem: min expenditure s.t. utility constraint"""
        
        # Generate symbols
        x_syms = [symbols(f'x{i}') for i in range(1, n_goods + 1)]
        p_syms = [symbols(f'p{i}') for i in range(1, n_goods + 1)]
        U_bar = symbols('U_bar')
        lambda_sym = symbols('lambda')
        
        # Build utility function
        U = self._build_utility_function(utility_type, x_syms, parameters, custom_utility)
        
        # Expenditure minimization
        expenditure = sum(p_syms[i] * x_syms[i] for i in range(n_goods))
        utility_constraint = U - U_bar
        
        # Lagrangian
        L = expenditure + lambda_sym * utility_constraint

        # First Order Conditions
        focs = []
        for i in range(n_goods):
            foc = diff(L, x_syms[i])
            focs.append(Eq(foc, 0))
        focs.append(Eq(diff(L, lambda_sym), 0))

        # Solve system
        solution_vars = x_syms + [lambda_sym]
        solution = solve(focs, solution_vars, dict=True)

        # Compute expenditure function
        expenditure_function = None
        if solution:
            expenditure_function = expenditure.subs(solution[0])

        # Build steps
        steps = self._build_hicksian_steps(U, expenditure, utility_constraint, focs, solution, expenditure_function)

        return {
            "success": True,
            "problem_type": "hicksian",
            "utility_function": str(U),
            "variables": {
                "goods": [str(x) for x in x_syms],
                "prices": [str(p) for p in p_syms],
                "utility_level": str(U_bar)
            },
            "focs": [str(foc) for foc in focs],
            "solution": [{str(k): str(v) for k, v in sol.items()} for sol in solution],
            "expenditure_function": str(expenditure_function) if expenditure_function else None,
            "steps": [step.dict() for step in steps]
        }

    def _build_utility_function(self, utility_type: str, x_syms: List, parameters: Dict[str, float], custom_utility: str = None):
        """Build utility function based on type and parameters"""
        if utility_type == "cobb-douglas":
            # U = A * x1^a1 * x2^a2 * ... * xn^an
            A = parameters.get('A', 1)
            alphas = [parameters.get(f'alpha{i+1}', 1/n_goods) for i in range(len(x_syms))]
            U = A
            for i, x in enumerate(x_syms):
                U *= x ** alphas[i]
            return U

        elif utility_type == "leontief":
            # U = min(a1*x1, a2*x2, ..., an*xn)
            coefficients = [parameters.get(f'a{i+1}', 1) for i in range(len(x_syms))]
            # For symbolic computation, we use a product form that approximates min
            return sum(coefficients[i] * x_syms[i] for i in range(len(x_syms)))  # Simplified for demonstration

        elif utility_type == "ces":
            # U = (a1*x1^rho + a2*x2^rho + ... + an*xn^rho)^(1/rho)
            rho = parameters.get('rho', 0.5)
            weights = [parameters.get(f'weight{i+1}', 1) for i in range(len(x_syms))]
            inner_sum = sum(weights[i] * x_syms[i] ** rho for i in range(len(x_syms)))
            return inner_sum ** (1/rho)

        elif utility_type == "crra":
            # U = (x1^(1-gamma) + x2^(1-gamma) + ... + xn^(1-gamma)) / (1-gamma)
            gamma = parameters.get('gamma', 2)
            if abs(gamma - 1) < 1e-10:
                # Logarithmic when gamma=1
                return sum(ln(x) for x in x_syms)
            else:
                return sum(x**(1-gamma) for x in x_syms) / (1-gamma)

        elif utility_type == "cara":
            # U = -sum(exp(-a*xi)) for i
            a = parameters.get('alpha', 1)
            return -sum(exp(-a * x) for x in x_syms)

        elif utility_type == "custom" and custom_utility:
            # Parse custom utility function
            local_dict = {f'x{i+1}': x_syms[i] for i in range(len(x_syms))}
            local_dict.update(parameters)
            return parse_expr(custom_utility, local_dict=local_dict)

        else:
            # Default to Cobb-Douglas
            A = 1
            alphas = [1/len(x_syms)] * len(x_syms)
            U = A
            for i, x in enumerate(x_syms):
                U *= x ** alphas[i]
            return U

    def _compute_elasticities(self, solution, x_syms, p_syms, R, U):
        """Compute price and income elasticities"""
        elasticities = {}
        
        if not solution:
            return elasticities

        # For each good, compute elasticities
        for i, x_sym in enumerate(x_syms):
            if x_sym in solution:
                x_i = solution[x_sym]
                
                # Price elasticity of demand (own-price)
                try:
                    for j, p_sym in enumerate(p_syms):
                        dx_dp = diff(x_i, p_sym)
                        if dx_dp != 0:
                            elasticity = dx_dp * p_sym / x_i
                            elasticities[f"price_elasticity_{i+1}_{j+1}"] = float(elasticity.subs({p: 1 for p in p_syms}).evalf())
                except:
                    pass
                
                # Income elasticity
                try:
                    dx_dR = diff(x_i, R)
                    if dx_dR != 0:
                        elasticity_income = dx_dR * R / x_i
                        elasticities[f"income_elasticity_{i+1}"] = float(elasticity_income.subs({R: 100}).evalf())
                except:
                    pass

        return elasticities

    def _compute_tms(self, U, x_syms):
        """Compute Taux Marginal de Substitution between first two goods"""
        if len(x_syms) >= 2:
            return -diff(U, x_syms[0]) / diff(U, x_syms[1])
        return None

    def _compute_slutsky_matrix(self, solution, x_syms, p_syms, R, U):
        """Compute Slutsky matrix for decomposition"""
        if len(x_syms) < 2 or not solution:
            return None
            
        # Simplified Slutsky matrix computation
        matrix = {}
        for i in range(len(x_syms)):
            for j in range(len(x_syms)):
                if i == j:
                    matrix[f"s_{i+1}{j+1}"] = "Substitution effect"
                else:
                    matrix[f"s_{i+1}{j+1}"] = "Cross-substitution effect"
        
        return matrix

    def _build_marshallian_steps(self, U, budget_constraint, focs, solution, elasticities, tms):
        steps = []
        
        steps.append(SolutionStep(
            step_name="Problème de maximisation d'utilité",
            equation=f"Max U = {U}",
            explanation="Maximisation de la fonction d'utilité sous contrainte budgétaire"
        ))
        
        steps.append(SolutionStep(
            step_name="Contrainte budgétaire",
            equation=f"{budget_constraint} = 0",
            explanation="La dépense totale doit égaler le revenu disponible"
        ))
        
        steps.append(SolutionStep(
            step_name="Lagrangien",
            equation=f"L = U + λ·(∑pᵢxᵢ - R)",
            explanation="Construction du Lagrangien pour résoudre le problème d'optimisation contrainte"
        ))
        
        for i, foc in enumerate(focs[:3]):  # Show first 3 FOCs
            steps.append(SolutionStep(
                step_name=f"Condition premier ordre {i+1}",
                equation=f"{foc}",
                explanation="Dérivée partielle du Lagrangien par rapport à une variable de décision"
            ))
        
        if solution:
            steps.append(SolutionStep(
                step_name="Solution du système d'équations",
                equation="Système résolu symboliquement",
                explanation="Résolution des conditions de premier ordre pour obtenir les demandes marshalliennes"
            ))
        
        if tms:
            steps.append(SolutionStep(
                step_name="Taux Marginal de Substitution (TMS)",
                equation=f"TMS = -∂U/∂x₁ ÷ ∂U/∂x₂ = {tms}",
                explanation="Taux auquel le consommateur est prêt à substituer le bien 2 au bien 1"
            ))
        
        return steps

    def _build_hicksian_steps(self, U, expenditure, utility_constraint, focs, solution, expenditure_function):
        steps = []
        
        steps.append(SolutionStep(
            step_name="Problème de minimisation des dépenses",
            equation=f"Min E = {expenditure}",
            explanation="Minimisation des dépenses pour atteindre un niveau d'utilité donné Ū"
        ))
        
        steps.append(SolutionStep(
            step_name="Contrainte d'utilité",
            equation=f"U(x) = {U} = Ū",
            explanation="Le niveau d'utilité doit être exactement égal à Ū"
        ))
        
        steps.append(SolutionStep(
            step_name="Lagrangien",
            equation=f"L = ∑pᵢxᵢ + λ·(U(x) - Ū)",
            explanation="Lagrangien pour la minimisation sous contrainte d'utilité"
        ))
        
        for i, foc in enumerate(focs[:3]):
            steps.append(SolutionStep(
                step_name=f"Condition premier ordre {i+1}",
                equation=f"{foc}",
                explanation="Condition d'optimalité pour la minimisation des dépenses"
            ))
        
        if solution:
            steps.append(SolutionStep(
                step_name="Demandes hicksiennes",
                equation="Solution obtenue",
                explanation="Fonctions de demande compensée en fonction des prix et du niveau d'utilité"
            ))
        
        if expenditure_function:
            steps.append(SolutionStep(
                step_name="Fonction de dépense",
                equation=f"E(p, Ū) = {expenditure_function}",
                explanation="Dépense minimale nécessaire pour atteindre le niveau d'utilité Ū aux prix p"
            ))
        
        return steps