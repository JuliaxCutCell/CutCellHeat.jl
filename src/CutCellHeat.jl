module CutCellHeat

# Modules
using CutCellMesh
using CutCellGeometry
using LinearAlgebra
using SparseArrays

# Files
include("operators.jl")

export ẟ_m, ẟ_p, Σ_m, Σ_p, operators_2D_3D
export build_matrix_G, build_matrix_H, build_matrix_minus_GTHT, div_op, grad_op, pseudo_inverse
# Write your package code here.


end # module