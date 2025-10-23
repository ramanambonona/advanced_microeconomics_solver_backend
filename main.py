from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io

# Import des solveurs
from .static_solver import StaticSolver
from .dynamic_solver import DynamicSolver
from .stochastic_solver import StochasticSolver

app = FastAPI(
    title="Advanced Microeconomics Solver API",
    version="1.0.0",
    description="API backend pour le solver symbolique d'optimisation microéconomique",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS - Autoriser le frontend déployé séparément
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Dev frontend
        "http://127.0.0.1:3000",  # Dev frontend
        "https://advanced-microeconomics-solver.vercel.app",  # Production frontend
        "https://*.vercel.app",  # Tous les sous-domaines Vercel
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Modèles Pydantic pour les requêtes
class StaticOptions(BaseModel):
    compute_elasticities: bool = True
    compute_tms: bool = True
    show_steps: bool = True
    graphical_output: bool = True

class StaticProblem(BaseModel):
    problem_type: str  # "marshallian" or "hicksian"
    utility_type: str  # "cobb-douglas", "leontief", "ces", "crra", "cara", "custom"
    n_goods: int
    parameters: Dict[str, float]
    constraints: List[str] = []
    custom_utility: Optional[str] = None
    options: Optional[StaticOptions] = None

class DynamicProblem(BaseModel):
    problem_type: str  # "discrete" or "continuous"
    T: Union[int, str]  # time periods or "infinite"
    utility_expr: str
    constraint_expr: str
    parameters: Dict[str, float]
    beta: float = 0.96

class StochasticProblem(BaseModel):
    problem_type: str  # "markov", "vector"
    T: int
    S: int  # states
    P: List[List[float]]  # transition matrix
    utility_type: str
    constraints: List[str]
    parameters: Dict[str, float]

class MonteCarloRequest(BaseModel):
    T: int
    S: int
    P: List[List[float]]
    policy_function: Optional[str] = None
    parameters: Dict[str, float] = {}
    n_simulations: int = 1000

class SymbolGenerationRequest(BaseModel):
    prefix: str
    n: int
    start: int = 1
    assume_positive: bool = True

class PlotRequest(BaseModel):
    expression: str
    range_start: float = 0
    range_end: float = 10
    x_label: str = "x"
    y_label: str = "f(x)"
    title: str = "Graph of function"

# Routes de l'API
@app.get("/")
async def read_root():
    """Page d'accueil de l'API backend"""
    return {
        "message": "Advanced Microeconomics Solver API Backend",
        "version": "1.0.0",
        "status": "online",
        "modules": ["statique", "dynamique", "stochastique"],
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "static": "/solve/static",
            "dynamic": "/solve/dynamic", 
            "stochastic": "/solve/stochastic",
            "monte_carlo": "/monte-carlo",
            "symbols": "/generate/symbols",
            "plot": "/plot/function",
            "modules": "/api/modules",
            "utility_functions": "/api/utility-functions"
        }
    }

@app.get("/health")
async def health():
    """Endpoint de santé de l'application"""
    return {
        "status": "ok", 
        "service": "Advanced Microeconomics Solver Backend",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.post("/solve/static")
async def solve_static(problem: StaticProblem):
    """Solve static optimization problem of order n"""
    try:
        solver = StaticSolver()
        result = solver.solve(
            problem_type=problem.problem_type,
            utility_type=problem.utility_type,
            n_goods=problem.n_goods,
            parameters=problem.parameters,
            constraints=problem.constraints,
            custom_utility=problem.custom_utility,
            options=problem.options.dict() if problem.options else {}
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur dans le module statique: {str(e)}"
        )

@app.post("/solve/dynamic")
async def solve_dynamic(problem: DynamicProblem):
    """Solve dynamic optimization problem of order T"""
    try:
        # Handle infinite horizon
        T = float('inf') if problem.T == "infinite" else int(problem.T)
        
        solver = DynamicSolver()
        result = solver.solve(
            problem_type=problem.problem_type,
            T=T,
            utility_expr=problem.utility_expr,
            constraint_expr=problem.constraint_expr,
            parameters=problem.parameters,
            beta=problem.beta
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur dans le module dynamique: {str(e)}"
        )

@app.post("/solve/stochastic")
async def solve_stochastic(problem: StochasticProblem):
    """Solve stochastic optimization problem"""
    try:
        solver = StochasticSolver()
        result = solver.solve(
            problem_type=problem.problem_type,
            T=problem.T,
            S=problem.S,
            P=problem.P,
            utility_type=problem.utility_type,
            constraints=problem.constraints,
            parameters=problem.parameters
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur dans le module stochastique: {str(e)}"
        )

@app.post("/monte-carlo")
async def run_monte_carlo(request: MonteCarloRequest):
    """Run Monte Carlo simulation for stochastic problems"""
    try:
        solver = StochasticSolver()
        result = solver.monte_carlo_simulation(
            T=request.T,
            S=request.S,
            P=request.P,
            policy_function=request.policy_function,
            parameters=request.parameters,
            n_simulations=request.n_simulations
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur dans la simulation Monte Carlo: {str(e)}"
        )

@app.post("/generate/symbols")
async def generate_symbols(request: SymbolGenerationRequest):
    """Generate symbolic variables x1, x2, ..., xn"""
    try:
        symbols = [f"{request.prefix}{i}" for i in range(request.start, request.start + request.n)]
        return JSONResponse(content={
            "symbols": symbols,
            "count": len(symbols),
            "prefix": request.prefix,
            "assume_positive": request.assume_positive
        })
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur de génération de symboles: {str(e)}"
        )

@app.post("/plot/function")
async def plot_function(request: PlotRequest):
    """Generate plot for a mathematical function"""
    try:
        x = sp.symbols('x')
        expr = sp.sympify(request.expression)
        f = sp.lambdify(x, expr, 'numpy')
        
        x_vals = np.linspace(request.range_start, request.range_end, 400)
        y_vals = f(x_vals)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=request.expression)
        plt.grid(True, alpha=0.3)
        plt.xlabel(request.x_label)
        plt.ylabel(request.y_label)
        plt.title(request.title)
        plt.legend()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Return base64 encoded image for JSON response
        img_base64 = __import__('base64').b64encode(buf.getvalue()).decode()
        return JSONResponse(content={
            "success": True,
            "plot": f"data:image/png;base64,{img_base64}",
            "expression": request.expression
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur de génération de graphique: {str(e)}"
        )

@app.get("/api/modules")
async def get_modules():
    """Retourne la liste des modules disponibles"""
    return JSONResponse(content={
        "modules": [
            {
                "name": "statique",
                "description": "Optimisation microéconomique statique d'ordre n",
                "endpoint": "/solve/static",
                "problem_types": ["marshallian", "hicksian"],
                "utility_functions": ["cobb-douglas", "leontief", "ces", "crra", "cara", "custom"]
            },
            {
                "name": "dynamique", 
                "description": "Optimisation intertemporelle d'ordre T",
                "endpoint": "/solve/dynamic",
                "problem_types": ["discrete", "continuous"],
                "horizons": ["finite", "infinite"]
            },
            {
                "name": "stochastique",
                "description": "Optimisation sous incertitude",
                "endpoint": "/solve/stochastic",
                "problem_types": ["markov", "vector"],
                "methods": ["symbolic", "monte_carlo"]
            }
        ]
    })

@app.get("/api/utility-functions")
async def get_utility_functions():
    """Retourne la liste des fonctions d'utilité supportées"""
    return JSONResponse(content={
        "utility_functions": [
            {
                "name": "cobb-douglas",
                "formula": "U = A * x₁^α₁ * x₂^α₂ * ... * xₙ^αₙ",
                "parameters": ["A", "α₁", "α₂", "..."],
                "properties": ["élasticité de substitution = 1", "demandes proportionnelles au revenu"]
            },
            {
                "name": "leontief",
                "formula": "U = min(x₁/a₁, x₂/a₂, ..., xₙ/aₙ)", 
                "parameters": ["a₁", "a₂", "..."],
                "properties": ["compléments parfaits", "pas de substitution"]
            },
            {
                "name": "ces",
                "formula": "U = (α₁x₁^ρ + α₂x₂^ρ + ... + αₙxₙ^ρ)^(1/ρ)",
                "parameters": ["ρ", "α₁", "α₂", "..."],
                "properties": ["élasticité de substitution constante", "généralise Cobb-Douglas et Leontief"]
            },
            {
                "name": "crra",
                "formula": "U = ∑(xᵢ^(1-γ))/(1-γ) pour γ ≠ 1, U = ∑ln(xᵢ) pour γ = 1",
                "parameters": ["γ"],
                "properties": ["aversion relative au risque constante", "utilité espérée"]
            },
            {
                "name": "cara", 
                "formula": "U = -∑exp(-αxᵢ)",
                "parameters": ["α"],
                "properties": ["aversion absolue au risque constante", "utilité espérée"]
            },
            {
                "name": "custom",
                "formula": "Définie par l'utilisateur",
                "parameters": "Variables libres",
                "properties": ["flexibilité totale", "expression symbolique"]
            }
        ]
    })

# Endpoint de test des solveurs
@app.get("/api/test")
async def test_solvers():
    """Endpoint pour tester les solveurs"""
    try:
        # Test du solveur statique
        static_solver = StaticSolver()
        static_test = static_solver.solve(
            problem_type="marshallian",
            utility_type="cobb-douglas",
            n_goods=2,
            parameters={"alpha1": 0.3, "alpha2": 0.7},
            constraints=[]
        )
        
        # Test du solveur dynamique
        dynamic_solver = DynamicSolver()
        dynamic_test = dynamic_solver.solve(
            problem_type="discrete",
            T=3,
            utility_expr="log(c)",
            constraint_expr="k_next - (1-0.08)*k - k**0.33 + c",
            parameters={"delta": 0.08, "alpha": 0.33},
            beta=0.96
        )
        
        return JSONResponse(content={
            "static_solver": static_test.get("success", False),
            "dynamic_solver": dynamic_test.get("success", False),
            "status": "all_solvers_operational"
        })
        
    except Exception as e:
        return JSONResponse(content={
            "error": str(e),
            "status": "solvers_test_failed"
        }, status_code=500)

# Gestion des erreurs globales
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du serveur",
            "error": str(exc),
            "endpoint": str(request.url)
        }
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint non trouvé",
            "endpoint": str(request.url),
            "available_endpoints": [
                "/",
                "/health", 
                "/solve/static",
                "/solve/dynamic",
                "/solve/stochastic",
                "/api/modules",
                "/api/utility-functions",
                "/docs"
            ]
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Erreur de validation des données",
            "error": str(exc)
        }
    )
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

