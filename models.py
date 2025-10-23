from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from enum import Enum

class UtilityType(str, Enum):
    COBB_DOUGLAS = "cobb-douglas"
    LEONTIEF = "leontief"
    CES = "ces"
    CRRA = "crra"
    CARA = "cara"
    CUSTOM = "custom"

class ProblemType(str, Enum):
    MARSHALLIAN = "marshallian"
    HICKSIAN = "hicksian"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MARKOV = "markov"
    VECTOR = "vector"
    
class StaticOptions(BaseModel):
    compute_elasticities: bool = True
    compute_tms: bool = True
    show_steps: bool = True
    graphical_output: bool = True

class StaticProblem(BaseModel):
    problem_type: str
    utility_type: str
    n_goods: int
    parameters: Dict[str, float]
    constraints: List[str] = []
    custom_utility: Optional[str] = None
    options: Optional[StaticOptions] = None

class SolutionStep(BaseModel):
    step_name: str
    equation: str
    explanation: str

class OptimizationResult(BaseModel):
    success: bool
    objective: str
    variables: List[str]
    focs: List[str]  # First Order Conditions
    solution: Dict[str, str]
    steps: List[SolutionStep]
    plots: List[str]  # Base64 encoded plots
    elasticities: Dict[str, float]
    tms: Optional[str] = None  # Taux Marginal de Substitution
