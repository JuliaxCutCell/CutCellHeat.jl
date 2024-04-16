module CutCellHeat

# Modules
using CutCellMesh
using CutCellGeometry
using LinearAlgebra
using SparseArrays

# Files
include("operators.jl")

# Export functions
export ẟ_m, ẟ_p, Σ_m, Σ_p, operators_2D_3D
export build_matrix_G, build_matrix_H, build_matrix_minus_GTHT, div_op, grad_op, pseudo_inverse, I_gamma

end # module