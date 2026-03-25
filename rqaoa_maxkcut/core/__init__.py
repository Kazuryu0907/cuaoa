"""
rqaoa_maxkcut.core
==================
Core modules for Level-1 RQAOA on MAX-k-CUT.

Modules
-------
hamiltonian   : MaxKCutHamiltonian — cost function h_{uv}(a) and ĥ_{uv}(b)
density_matrix: DensityMatrixSimulator — 2-body marginal ρ_{uv}
angle_opt     : AngleOptimizer — analytic β-optimisation for k=3
rqaoa         : RQAOA1Solver — full Level-1 RQAOA loop
"""
from .hamiltonian import MaxKCutHamiltonian
from .density_matrix import DensityMatrixSimulator
from .angle_opt import AngleOptimizer
from .rqaoa import RQAOA1Solver

__all__ = [
    "MaxKCutHamiltonian",
    "DensityMatrixSimulator",
    "AngleOptimizer",
    "RQAOA1Solver",
]
