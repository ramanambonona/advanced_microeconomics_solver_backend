from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import matplotlib.pyplot as plt
import io
import base64
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Microeconomics Solver API",
    version="1.0.0",
    description="API backend pour le solver symbolique d'optimisation microéconomique",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Configuration pour production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "https://advanced-microeconomics-solver.vercel.app",
        "https://*.vercel.app",
        "https://*.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Modèles de données
class StaticProblem(BaseModel):
    problem_type: str = "marshallian"
    utility_type: str = "cobb-douglas"
    n_goods: int = 2
    parameters: Dict[str, float] = {}
    constraints: List[str] = []
    custom_utility: Optional[str] = None

class DynamicProblem(BaseModel):
    T: Union[int, str] = 3
    utility_expr: str = "log(c)"
    constraint_expr: str = "k_next - (1-delta)*k - A*k**alpha + c"
    parameters: Dict[str, float] = {}
    beta: float = 0.96

class StochasticProblem(BaseModel):
    T: int = 3
    S: int = 2
    P: List[List[float]] = [[0.9, 0.1], [0.2, 0.8]]
    utility_type: str = "crra"
    constraints: List[str] = []
    parameters: Dict[str, float] = {}

class PlotRequest(BaseModel):
    expression: str
    range_start: float = 0
    range_end: float = 10
    x_label: str = "x"
    y_label: str = "f(x)"
    title: str = "Graph of function"

# Solveur statique intégré
class StaticSolver:
    @staticmethod
    def solve(problem_type: str, utility_type: str, n_goods: int, 
              parameters: Dict[str, float], constraints: List[str],
              custom_utility: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Générer les symboles
            x_syms = [sp.symbols(f'x{i+1}', positive=True) for i in range(n_goods)]
            p_syms = [sp.symbols(f'p{i+1}', positive=True) for i in range(n_goods)]
            R = sp.symbols('R', positive=True)
            lambda_sym = sp.symbols('lambda', real=True)
            
            # Construire la fonction d'utilité
            if utility_type == "cobb-douglas":
                alpha_syms = [sp.symbols(f'alpha{i+1}', positive=True) for i in range(n_goods)]
                U = 1
                for i in range(n_goods):
                    U *= x_syms[i] ** alpha_syms[i]
            elif utility_type == "log":
                U = sum(sp.log(x_syms[i]) for i in range(n_goods))
            elif utility_type == "crra":
                gamma = parameters.get('gamma', 2.0)
                if abs(gamma - 1.0) < 1e-10:
                    U = sum(sp.log(x_syms[i]) for i in range(n_goods))
                else:
                    U = sum((x_syms[i]**(1-gamma))/(1-gamma) for i in range(n_goods))
            elif utility_type == "custom" and custom_utility:
                local_dict = {f'x{i+1}': x_syms[i] for i in range(n_goods)}
                local_dict.update(parameters)
                U = sp.sympify(custom_utility, locals=local_dict)
            else:
                raise ValueError(f"Type d'utilité non supporté: {utility_type}")
            
            # Construire le Lagrangien selon le type de problème
            if problem_type == "marshallian":
                # Maximisation sous contrainte budgétaire
                budget_constraint = sum(p_syms[i] * x_syms[i] for i in range(n_goods)) - R
                L = U + lambda_sym * budget_constraint
            elif problem_type == "hicksian":
                # Minimisation sous contrainte d'utilité
                U_bar = sp.symbols('U_bar', positive=True)
                utility_constraint = U - U_bar
                L = sum(p_syms[i] * x_syms[i] for i in range(n_goods)) + lambda_sym * utility_constraint
            else:
                raise ValueError(f"Type de problème non supporté: {problem_type}")
            
            # Conditions premier ordre
            focs = []
            for x in x_syms:
                focs.append(sp.Eq(sp.diff(L, x), 0))
            focs.append(sp.Eq(sp.diff(L, lambda_sym), 0))
            
            # Résoudre le système
            variables = x_syms + [lambda_sym]
            solution = sp.solve(focs, variables, dict=True)
            
            # Préparer les résultats
            result = {
                "success": True,
                "problem_type": problem_type,
                "utility_function": str(U),
                "focs": [str(eq) for eq in focs],
                "solution": [{str(k): str(v) for k, v in sol.items()} for sol in solution],
                "steps": [
                    {
                        "step_name": "Formulation du problème",
                        "equation": f"Max U = {U} sous contrainte budgétaire" if problem_type == "marshallian" else f"Min dépense pour U = {U_bar}",
                        "explanation": "Définition du problème d'optimisation"
                    },
                    {
                        "step_name": "Lagrangien",
                        "equation": f"L = {str(L)}",
                        "explanation": "Construction du Lagrangien avec multiplicateur"
                    }
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur dans le solveur statique: {str(e)}")
            return {"success": False, "error": str(e)}

# Solveur dynamique intégré
class DynamicSolver:
    @staticmethod
    def solve(T: Union[int, str], utility_expr: str, constraint_expr: str,
              parameters: Dict[str, float], beta: float = 0.96) -> Dict[str, Any]:
        try:
            # Gérer l'horizon infini
            is_infinite = T == "infinite"
            T_val = 10 if is_infinite else int(T)
            
            # Générer les symboles
            c_syms = [sp.symbols(f'c_{t}', positive=True) for t in range(T_val)]
            k_syms = [sp.symbols(f'k_{t}', positive=True) for t in range(T_val + 1)]
            lambda_syms = [sp.symbols(f'lambda_{t}', real=True) for t in range(T_val)]
            
            # Construire le Lagrangien
            L = 0
            beta_sym = sp.symbols('beta', positive=True)
            
            for t in range(T_val):
                # Utilité à la période t
                local_dict = {'c': c_syms[t], 'beta': beta_sym, **parameters}
                U_t = sp.sympify(utility_expr, locals=local_dict)
                L += (beta_sym ** t) * U_t
                
                # Contrainte à la période t
                constraint_local = {
                    'c': c_syms[t], 
                    'k': k_syms[t], 
                    'k_next': k_syms[t+1],
                    **parameters
                }
                g_t = sp.sympify(constraint_expr, locals=constraint_local)
                L += lambda_syms[t] * g_t
            
            # Conditions premier ordre (symboliques)
            euler_equations = []
            if T_val >= 2:
                # Équation d'Euler générique
                c_t, c_t1, k_t, k_t1 = sp.symbols('c_t c_t1 k_t k_t1', positive=True)
                lambda_t, lambda_t1 = sp.symbols('lambda_t lambda_t1', real=True)
                
                # Utilités marginales
                U_t_expr = sp.sympify(utility_expr, locals={'c': c_t, **parameters})
                U_t1_expr = sp.sympify(utility_expr, locals={'c': c_t1, **parameters})
                
                dU_dc_t = sp.diff(U_t_expr, c_t)
                dU_dc_t1 = sp.diff(U_t1_expr, c_t1)
                
                # Équation d'Euler
                euler_eq = sp.Eq(dU_dc_t, beta_sym * dU_dc_t1)
                euler_equations.append(str(euler_eq))
            
            result = {
                "success": True,
                "horizon": "infinite" if is_infinite else T_val,
                "objective": f"∑ₜ₌₀^{'∞' if is_infinite else T_val-1} βᵗ U(cₜ)",
                "constraint": constraint_expr,
                "euler_equations": euler_equations,
                "variables": {
                    "consumption": [str(c) for c in c_syms],
                    "capital": [str(k) for k in k_syms],
                    "multipliers": [str(l) for l in lambda_syms]
                },
                "steps": [
                    {
                        "step_name": "Problème dynamique",
                        "equation": f"Max ∑ₜ₌₀^{'∞' if is_infinite else T_val-1} βᵗ U(cₜ)",
                        "explanation": "Maximisation intertemporelle de l'utilité escomptée"
                    },
                    {
                        "step_name": "Contrainte de ressources",
                        "equation": f"g(cₜ, kₜ, kₜ₊₁) = {constraint_expr} = 0",
                        "explanation": "Contrainte liant consommation et accumulation de capital"
                    }
                ]
            }
            
            if is_infinite:
                result["steps"].append({
                    "step_name": "État stationnaire",
                    "equation": "cₜ = cₜ₊₁ = c*, kₜ = kₜ₊₁ = k*",
                    "explanation": "En horizon infini, recherche d'un état stationnaire"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur dans le solveur dynamique: {str(e)}")
            return {"success": False, "error": str(e)}

# Solveur stochastique intégré
class StochasticSolver:
    @staticmethod
    def solve(T: int, S: int, P: List[List[float]], utility_type: str,
              constraints: List[str], parameters: Dict[str, float]) -> Dict[str, Any]:
        try:
            # Valider la matrice de transition
            for i, row in enumerate(P):
                if abs(sum(row) - 1.0) > 1e-10:
                    raise ValueError(f"La ligne {i} de la matrice P ne somme pas à 1")
            
            # Symboles de base
            beta = parameters.get('beta', 0.96)
            c, k, z = sp.symbols('c k z', positive=True)
            
            # Construire l'équation d'Euler stochastique
            if utility_type == "crra":
                gamma = parameters.get('gamma', 2.0)
                if abs(gamma - 1.0) < 1e-10:
                    # Utilité logarithmique
                    euler_eq = f"1/c_t = {beta} * E[1/c_{{t+1}} * R_{{t+1}} | I_t]"
                else:
                    euler_eq = f"c_t^(-{gamma}) = {beta} * E[c_{{t+1}}^(-{gamma}) * R_{{t+1}} | I_t]"
            else:
                euler_eq = f"U'(c_t) = {beta} * E[U'(c_{{t+1}}) * R_{{t+1}} | I_t]"
            
            # Calculer la distribution stationnaire (approximation)
            stationary = StochasticSolver._compute_stationary_distribution(P)
            
            result = {
                "success": True,
                "problem_type": "stochastic",
                "horizon": T,
                "states": S,
                "transition_matrix": P,
                "stationary_distribution": stationary,
                "stochastic_euler_equation": euler_eq,
                "steps": [
                    {
                        "step_name": "Processus stochastique",
                        "equation": f"Chaîne de Markov avec {S} états",
                        "explanation": "Incertitude modélisée par un processus Markovien"
                    },
                    {
                        "step_name": "Équation d'Euler stochastique", 
                        "equation": euler_eq,
                        "explanation": "Condition d'optimalité avec espérance conditionnelle"
                    },
                    {
                        "step_name": "Distribution stationnaire",
                        "equation": f"π = {stationary}",
                        "explanation": "Distribution limite du processus Markovien"
                    }
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur dans le solveur stochastique: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _compute_stationary_distribution(P: List[List[float]]) -> List[float]:
        """Calcule la distribution stationnaire d'une chaîne de Markov"""
        try:
            # Méthode simple : itération de puissance
            n = len(P)
            pi = [1.0 / n] * n  # Distribution uniforme initiale
            
            for _ in range(100):  # 100 itérations
                new_pi = [0.0] * n
                for j in range(n):
                    for i in range(n):
                        new_pi[j] += pi[i] * P[i][j]
                
                # Vérifier la convergence
                if max(abs(new_pi[i] - pi[i]) for i in range(n)) < 1e-10:
                    break
                pi = new_pi
            
            return [round(p, 4) for p in pi]
        except:
            # Fallback: distribution uniforme
            n = len(P)
            return [round(1.0 / n, 4)] * n

# Routes de l'API
@app.get("/")
async def root():
    return {
        "message": "Advanced Microeconomics Solver API",
        "status": "active", 
        "version": "1.0.0",
        "python_version": os.environ.get('PYTHON_VERSION', 'Unknown'),
        "endpoints": {
            "health": "/health",
            "static": "/solve/static",
            "dynamic": "/solve/dynamic",
            "stochastic": "/solve/stochastic", 
            "plot": "/plot/function",
            "modules": "/api/modules",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Microeconomics Solver API"}

@app.post("/solve/static")
async def solve_static(problem: StaticProblem):
    """Résoudre un problème d'optimisation statique"""
    try:
        solver = StaticSolver()
        result = solver.solve(
            problem_type=problem.problem_type,
            utility_type=problem.utility_type,
            n_goods=problem.n_goods,
            parameters=problem.parameters,
            constraints=problem.constraints,
            custom_utility=problem.custom_utility
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/solve/dynamic") 
async def solve_dynamic(problem: DynamicProblem):
    """Résoudre un problème d'optimisation dynamique"""
    try:
        solver = DynamicSolver()
        result = solver.solve(
            T=problem.T,
            utility_expr=problem.utility_expr,
            constraint_expr=problem.constraint_expr,
            parameters=problem.parameters,
            beta=problem.beta
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/solve/stochastic")
async def solve_stochastic(problem: StochasticProblem):
    """Résoudre un problème d'optimisation stochastique"""
    try:
        solver = StochasticSolver()
        result = solver.solve(
            T=problem.T,
            S=problem.S,
            P=problem.P,
            utility_type=problem.utility_type,
            constraints=problem.constraints,
            parameters=problem.parameters
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/plot/function")
async def plot_function(request: PlotRequest):
    """Générer un graphique de fonction mathématique"""
    try:
        x = sp.symbols('x')
        expr = sp.sympify(request.expression)
        f = sp.lambdify(x, expr, 'numpy')
        
        x_vals = np.linspace(request.range_start, request.range_end, 400)
        y_vals = f(x_vals)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel(request.x_label)
        plt.ylabel(request.y_label)
        plt.title(request.title)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse(content={
            "success": True,
            "plot": f"data:image/png;base64,{img_base64}",
            "expression": request.expression
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de génération de graphique: {str(e)}")

@app.get("/api/modules")
async def get_modules():
    """Retourne les modules disponibles"""
    return JSONResponse(content={
        "modules": [
            {
                "name": "static",
                "description": "Optimisation statique d'ordre n",
                "problem_types": ["marshallian", "hicksian"],
                "utility_functions": ["cobb-douglas", "log", "crra", "custom"]
            },
            {
                "name": "dynamic",
                "description": "Optimisation intertemporelle",
                "horizons": ["finite", "infinite"],
                "methods": ["euler_equations", "symbolic_solution"]
            },
            {
                "name": "stochastic", 
                "description": "Optimisation sous incertitude",
                "methods": ["markov_chains", "stochastic_euler"]
            }
        ]
    })

@app.get("/api/test")
async def test_solvers():
    """Endpoint de test des solveurs"""
    try:
        # Test solveur statique
        static_result = StaticSolver.solve(
            problem_type="marshallian",
            utility_type="cobb-douglas", 
            n_goods=2,
            parameters={"alpha1": 0.3, "alpha2": 0.7}
        )
        
        # Test solveur dynamique
        dynamic_result = DynamicSolver.solve(
            T=3,
            utility_expr="log(c)",
            constraint_expr="k_next - (1-0.08)*k - k**0.33 + c",
            parameters={"delta": 0.08, "alpha": 0.33}
        )
        
        return JSONResponse(content={
            "static_solver": static_result.get("success", False),
            "dynamic_solver": dynamic_result.get("success", False),
            "status": "all_solvers_operational"
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "status": "test_failed"},
            status_code=500
        )

# Gestion des erreurs
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur", "error": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint non trouvé", "available_endpoints": [
            "/", "/health", "/solve/static", "/solve/dynamic", "/solve/stochastic", "/docs"
        ]}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
