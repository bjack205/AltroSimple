abstract type Cone end
struct IdentityCone <: Cone end
struct ZeroCone <: Cone end
struct NegativeOrthant <: Cone end
struct SecondOrderCone <: Cone end

struct SoftInequality <: Cone
    beta::Float64
end


dualcone(::IdentityCone) = ZeroCone()
dualcone(::ZeroCone) = IdentityCone()
dualcone(::NegativeOrthant) = NegativeOrthant()
dualcone(::SecondOrderCone) = SecondOrderCone()
dualcone(cone::SoftInequality) = SoftInequality(cone.beta)

function softplus(x; β=50)
    s = sign(β)
    s*x > s*10/β && return x
    s*x < -s*10/β && return zero(x) 
    log(1 + exp(β*x)) / β
end
softminus(x; β=50) = softplus(x, β=-β)
sigmoid(x; β=50) = 1/(1+exp(-x*β))
∇sigmoid(x; β=50) = β*sigmoid(x; β)*sigmoid(-x; β)

in_cone(::IdentityCone, x, tol) = true
in_cone(::ZeroCone, x, tol) = all(x->abs(x) <= tol, x)
in_cone(::NegativeOrthant, x, tol) = all(x->x <= tol, x)
in_cone(::SecondOrderCone, x, tol) = norm(view(x,1:length(x)-1)) <= (x[end] + tol)

in_polar_recession_cone(::ZeroCone, x, tol) = all(x->abs(x) <= tol, x)
in_polar_recession_cone(::NegativeOrthant, x, tol) = all(x->x >= -tol, x)
in_polar_recession_cone(::SecondOrderCone, x, tol) = norm(@view x[1:end-1]) <= (tol - x[end])

has_linear_projection(::ZeroCone) = true
has_linear_projection(::IdentityCone) = true
has_linear_projection(::NegativeOrthant) = true
has_linear_projection(::SecondOrderCone) = false 

projection(::IdentityCone, x) = x
projection(::ZeroCone, x) = zero(x)
projection(::NegativeOrthant, x) = min.(0,x)
projection(cone::SoftInequality, x) = softminus.(x, β=cone.beta)

function projection(::SecondOrderCone, x)
    n = length(x)
    s = x[end]
    v = view(x,1:n-1)
    a = norm(v)
    if a <= -s  # below the cone
        return zero(x)
    elseif a <= s  # in the code
        return x
    elseif a >= abs(s)  # outside the cone
        return 0.5 * (1 + s/a) * [v; a] 
    end
end

∇projection(::IdentityCone, x) = I
∇projection(::ZeroCone, x) = zeros(length(x), length(x))
function ∇projection(::NegativeOrthant, x)
    n = length(x)
    J = Diagonal(ones(n))
    for i = 1:n
        J[i,i] = x[i] <= 0
    end
    J
end
function ∇projection(::SecondOrderCone, x)
    n = length(x)
    J = zeros(eltype(x),n,n)
    s = x[end]
    v = view(x,1:n-1)
    a = norm(v)
    if a <= -s
        return J
    elseif a <= s
        J .= I(n)
        return J
    elseif a >= abs(s)
        c = 0.5 * (1 + s/a)

        # dvdv
        for i = 1:n-1, j = 1:n-1
            J[i,j] = -0.5*s/a^3 * v[i] * v[j]
            if i == j
                J[i,j] += c
            end
        end

        # dvds
        for i = 1:n-1
            J[i,n] = 0.5 * v[i] / a
        end

        # ds
        for i = 1:n-1
            J[n,i] = ((-0.5*s/a^2) + c/a) * v[i]
        end
        J[n,n] = 0.5 
        return J
    else
        error("Invalid second-order cone projection.")
    end
    return J
end

function ∇projection(cone::SoftInequality, x)
    n = length(x)
    J = Diagonal(ones(n))
    for i = 1:n
        J[i,i] = sigmoid(x[i], β=-cone.beta)
    end
    return J
end

∇²projection(::IdentityCone, x, b) = zeros(length(x), length(x))
∇²projection(::ZeroCone, x, b) = zeros(length(x), length(x))
∇²projection(::NegativeOrthant, x, b) = zeros(length(x), length(x))

function ∇²projection(::SecondOrderCone, x, b)
    n = length(x)
    s = x[end]
    v = view(x, 1:n-1)
    bv = view(b, 1:n-1)
    a = norm(v)
    bs = b[end]
    vbv = dot(v,bv)
    hess = zeros(n,n)
    if a <= -s
    elseif a <= s
    elseif a >= abs(s)
        n = n-1
        dvdv = view(hess, 1:n, 1:n)
        dvds = view(hess, 1:n, n+1)
        dsdv = view(hess, n+1, 1:n)
        @inbounds for i = 1:n
            hi = 0
            @inbounds for j = 1:n
                Hij = -v[i]*v[j] / a^2
                if i == j
                    Hij += 1
                end
                hi += Hij * bv[j]
            end
            dvds[i] = hi / 2a
            dsdv[i] = dvds[i]
            @inbounds for j = 1:i
                vij = v[i] * v[j]
                H1 = hi * v[j] * (-s/a^3)
                H2 = vij * (2*vbv) / a^4 - v[i] * bv[j] / a^2
                H3 = -vij / a^2
                if i == j
                    H2 -= vbv / a^2
                    H3 += 1
                end
                H2 *= s/a
                H3 *= bs/a
                dvdv[i,j] = (H1 + H2 + H3) / 2
                dvdv[j,i] = dvdv[i,j]
            end
        end
        hess[end,end] = 0
    end
    return hess
end

function ∇²projection(cone::SoftInequality, x, b)
    n = length(x)
    H = Diagonal(ones(n))
    for i = 1:n
        H.diag[i] = ∇sigmoid(-x[i], β=-cone.beta)*b[i]
    end
    H
end

function dx_hess_projection(::SecondOrderCone, x, b)
    n = length(x)
    s = x[end]
    v = view(x, 1:n-1)
    bv = view(b, 1:n-1)
    a = norm(v)
    bs = b[end]
    vbv = dot(v,bv)
    hess = zeros(n*n,n)

    if a <= -s
    elseif a <= s
    elseif a >= abs(s)
        n = n-1
        dvdv = view(hess, 1:n, 1:n)
        dvds = view(hess, 1:n, n+1)
        dsdv = view(hess, n+1, 1:n)
        @inbounds for i = 1:n
            hi = 0
            @inbounds for j = 1:n
                Hij = -v[i]*v[j] / a^2
                if i == j
                    Hij += 1
                end
                hi += Hij * bv[j]
            end
            dvds[i] = hi / 2a
            dsdv[i] = dvds[i]
            @inbounds for j = 1:i
                vij = v[i] * v[j]
                H1 = hi * v[j] * (-s/a^3)
                H2 = vij * (2*vbv) / a^4 - v[i] * bv[j] / a^2
                H3 = -vij / a^2
                if i == j
                    H2 -= vbv / a^2
                    H3 += 1
                end
                H2 *= s/a
                H3 *= bs/a
                dvdv[i,j] = (H1 + H2 + H3) / 2
                dvdv[j,i] = dvdv[i,j]
            end
        end
        hess[end,end] = 0
    end
    return hess
end