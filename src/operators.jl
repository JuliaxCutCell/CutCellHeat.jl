"""
    ẟ_m(n::Int)

Constructs a sparse matrix representing the discrete backward difference operator for a 1D grid with `n` points.

# Arguments
- `n::Int`: The number of grid points.

# Optional Arguments
- `periodicity::Bool=false`: Specifies whether the grid is periodic or not. Default is `false`.

# Returns
- `D::SparseMatrixCSC`: The discrete difference operator.

# Example
"""
function ẟ_m(n::Int, periodicity::Bool=false)
    D = spdiagm(0 => ones(n), -1 => ones(n-1))
    D[n, n] = 0.0
    if periodicity
        D[1, n-1] = -1.0
        D[n, 1] = 1.0
    end
    return D
end

"""
    ẟ_p(n::Int, periodicity::Bool=false)

Constructs a sparse matrix representing the discrete forward difference operator for a 1D grid with `n` points.

# Arguments
- `n::Int`: The size of the operator matrix.
- `periodicity::Bool=false`: Specifies whether the grid is periodic or not. Default is `false`.

# Returns
A sparse difference operator matrix `D` of size `n x n` with the following properties:
- `D[i, i] = 0.0` for `i = 1`
- `D[i, i] = -1.0` for `i = 2:n`
- `D[i, i-1] = 1.0` for `i = 2:n`

# Example
"""
function ẟ_p(n::Int, periodicity::Bool=false)
    D = spdiagm(0 => -ones(n), 1 => ones(n-1))
    D[n, n] = 0.0
    if periodicity
        D[1, n-1] = -1.0
        D[n, 1] = 1.0
    end
    return D
end

"""
    Σ_m(n::Int, periodicity::Bool=false)

Constructs a sparse matrix representing the discrete backward interpolation operator for a 1D grid with `n` points.

# Arguments
- `n::Int`: The size of the matrix.
- `periodicity::Bool=false`: Specifies whether the grid is periodic or not. Default is `false`.

# Returns
- `D`: The constructed diagonal matrix.

# Examples
"""
function Σ_m(n::Int, periodicity::Bool=false)
    D = 0.5 * spdiagm(0 => ones(n), -1 => ones(n-1))
    D[n, n] = 0.0
    if periodicity
        D[1, n-1] = 0.5
        D[n, 1] = 0.5
    end
    return D
end

"""
    Σ_p(n::Int, periodicity::Bool=false)

Constructs a sparse matrix representing the discrete forward interpolation operator for a 1D grid with `n` points.

# Arguments
- `n::Int`: The size of the matrix.
- `periodicity::Bool=false`: Specifies whether the grid is periodic or not. Default is `false`.

# Returns
A sparse matrix representing the discrete derivative operator Σ_p.

# Examples
"""
function Σ_p(n::Int, periodicity::Bool=false)
    D = 0.5 * spdiagm(0 => ones(n), 1 => ones(n-1))
    D[n, n] = 0.0
    if periodicity
        D[1, n-1] = 0.5
        D[n, 1] = 0.5
    end
    return D
end

"""
    operators_2D_3D(operator, periodicity::Tuple{Bool,Bool,Bool}, nx::Int, ny::Int, nz::Int=0)

Constructs the 2D or 3D operator matrices based on the given operator function and dimensions.

# Arguments
- `operator`: A function that takes an integer argument and returns a matrix representing the desired operator.
- `periodicity`: A tuple of booleans specifying the periodicity in the x, y, and z directions.
- `nx`: The number of grid points in the x-direction.
- `ny`: The number of grid points in the y-direction.
- `nz`: (optional) The number of grid points in the z-direction. Defaults to 0, indicating a 2D problem.

# Returns
- If `nz > 0`, returns a 3D array of operator matrices in the order [x direction, y direction, z direction].
- If `nz <= 0`, returns a 2D array of operator matrices in the order [x direction, y direction].

# Examples
"""
function operators_2D_3D(operator, periodicity::Tuple{Bool,Bool,Bool}, nx::Int, ny::Int, nz::Int=0)
    px, py, pz = periodicity
    I_nx = spdiagm(0 => ones(nx))
    I_ny = spdiagm(0 => ones(ny))
    I_nz = nz > 0 ? spdiagm(0 => ones(nz)) : nothing
    D_nx = operator(nx, px)
    D_ny = operator(ny, py)
    D_nz = nz > 0 ? operator(nz, pz) : nothing

    matrices_2D = [
        kron(I_ny, D_nx),  # x direction
        kron(D_ny, I_nx)   # y direction
    ]

    if nz > 0
        matrices_3D = [
            kron(I_nz, kron(I_ny, D_nx)),  # x direction
            kron(I_nz, kron(D_ny, I_nx)),  # y direction
            kron(D_nz, kron(I_ny, I_nx))   # z direction
        ]
        return matrices_3D
    else
        return matrices_2D
    end
end

"""
    build_matrix_G(Dx_minus, Dy_minus, Bx, By)

Builds the matrix G by vertically concatenating the product of Dx_minus and Bx with the product of Dy_minus and By.

# Arguments
- `Dx_minus::AbstractMatrix`: The Dx_minus matrix.
- `Dy_minus::AbstractMatrix`: The Dy_minus matrix.
- `Bx::AbstractMatrix`: The Bx matrix.
- `By::AbstractMatrix`: The By matrix.

# Returns
- `G::AbstractMatrix`: The resulting matrix G.

# Example
"""
function build_matrix_G(Dx_minus, Dy_minus, Bx, By)    
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    G = vcat(Dx_minus_Bx, Dy_minus_By)
    
    return G
end


"""
    build_matrix_H(Dx_minus, Dy_minus, Ax, Ay, Bx, By)

Constructs the matrix H using the given input matrices.

# Arguments
- `Dx_minus::Matrix`: The Dx_minus matrix.
- `Dy_minus::Matrix`: The Dy_minus matrix.
- `Ax::Matrix`: The Ax matrix.
- `Ay::Matrix`: The Ay matrix.
- `Bx::Matrix`: The Bx matrix.
- `By::Matrix`: The By matrix.

# Returns
- `H::Matrix`: The constructed matrix H.

# Example
"""
function build_matrix_H(Dx_minus, Dy_minus, Ax, Ay, Bx, By)
    Ax_Dx_minus = Ax * Dx_minus
    Ay_Dy_minus = Ay * Dy_minus
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    block1 = Ax_Dx_minus - Dx_minus_Bx
    block2 = Ay_Dy_minus - Dy_minus_By
    
    H = vcat(block1, block2)
    
    return H
end

"""
    build_matrix_minus_GTHT(Dx_plus, Dy_plus, Ax, Ay)

Builds the matrix `minus_GTHT` by multiplying `Dx_plus` with `Ax` and `Dy_plus` with `Ay`, and concatenating the results horizontally.

# Arguments
- `Dx_plus::AbstractMatrix`: The Dx_plus matrix.
- `Dy_plus::AbstractMatrix`: The Dy_plus matrix.
- `Ax::AbstractMatrix`: The Ax matrix.
- `Ay::AbstractMatrix`: The Ay matrix.

# Returns
- `minus_GTHT::AbstractMatrix`: The resulting matrix.

"""
function build_matrix_minus_GTHT(Dx_plus, Dy_plus, Ax, Ay)
    Dx_plus_Ax = Dx_plus * Ax
    Dy_plus_Ay = Dy_plus * Ay
    
    block1 = Dx_plus_Ax
    block2 = Dy_plus_Ay
    
    minus_GTHT = hcat(block1, block2)
    
    return minus_GTHT
end

"""
    grad_op(p_omega, p_gamma, Wdagger, G, H)

Compute the gradient operator using the given parameters.

# Arguments
- `p_omega`: The p_omega parameter.
- `p_gamma`: The p_gamma parameter.
- `Wdagger`: The Wdagger parameter.
- `G`: The G parameter.
- `H`: The H parameter.

# Returns
The computed gradient.

"""
function grad_op(p_omega, p_gamma, Wdagger, G, H)
    G_pw = G * p_omega
    H_pgamma = H * p_gamma
    grad = Wdagger * (G_pw + H_pgamma)
    return grad
end

"""
    div_op(q_omega, q_gamma, GT, minus_GTHT, HT, gradient=false)

Compute the divergence of a vector field.

# Arguments
- `q_omega`: Vector field for the vorticity component.
- `q_gamma`: Vector field for the gamma component.
- `GT`: Matrix representing the gradient transpose operator.
- `minus_GTHT`: Matrix representing the negative gradient transpose times H operator.
- `HT`: Matrix representing the H transpose operator.
- `gradient`: Boolean indicating whether to compute the gradient (default: `false`).

# Returns
- `divergence`: The computed divergence of the vector field.

"""
function div_op(q_omega, q_gamma, GT, minus_GTHT, HT, gradient::Bool=false)
    if gradient
        divergence = -GT * q_omega
    else
        div_qw = minus_GTHT * q_omega
        div_qg = HT * q_gamma
        divergence = div_qw + div_qg
    end
    return divergence
end

"""
    pseudo_inverse(W::SparseMatrixCSC)

Compute the pseudo-inverse of a sparse matrix `W` using the diagonal values.

# Arguments
- `W::SparseMatrixCSC`: The input sparse matrix.

# Returns
- `W_inv::SparseMatrixCSC`: The pseudo-inverse of `W`.

# Examples
"""
function pseudo_inverse(W::SparseMatrixCSC)
    diag_values = diag(W)
    diag_values = [val != 0 ? 1/val : 1 for val in diag_values]
    diag_values = float.(diag_values)
    W_inv = spdiagm(0 => diag_values)
    return W_inv
end