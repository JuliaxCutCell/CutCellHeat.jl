using CutCellHeat
using CutCellMesh
using CutCellGeometry

n=11
grid = CartesianGrid((10, 10), (1.0, 1.0))
mesh = generate_mesh(grid)
domain = ((minimum(mesh[1]), maximum(mesh[1])), (minimum(mesh[2]), maximum(mesh[2])))

identity = (x, y, t) -> (x, y)
circle = (x, y, _=0) -> sqrt((x-5)^2 + (y-5)^2) - 0.5
circle_sdf = SignedDistanceFunction(circle, identity, domain, false)

V, v_diag, bary, Ax, Ay = calculate_first_order_moments(circle_sdf.sdf_function, mesh)
w_diag, Bx, By, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle_sdf.sdf_function, mesh, bary)

Dxm, Dym = operators_2D_3D(ẟ_m, (false, false, false), n, n)
Dxp, Dyp = operators_2D_3D(ẟ_p, (false, false, false), n, n)

H = build_matrix_H(Dxm, Dym, Ax, Ay, Bx, By)
G = build_matrix_G(Dxm, Dym, Bx, By)
GT = G'
HT = H'

Wdagger = pseudo_inverse(w_diag)