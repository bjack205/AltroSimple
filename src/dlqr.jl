
struct dLQR{T}
    # Problem data
    Q::Vector{Matrix{T}}
    R::Vector{Matrix{T}}
    H::Vector{Matrix{T}}
    q::Vector{Vector{T}}
    r::Vector{Vector{T}}
    A::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    f::Vector{Vector{T}}

    # TVLQR Data
    Qxx::Vector{Matrix{T}}
    Quu::Vector{Matrix{T}}
    Qux::Vector{Matrix{T}}
    Qx::Vector{Vector{T}}
    Qu::Vector{Vector{T}}
    P::Vector{Matrix{T}}
    p::Vector{Vector{T}}
    K::Vector{Matrix{T}}
    d::Vector{Vector{T}}

    # Derivative data
    dP_dQ::Vector{Matrix{T}} 
    dP_dR::Vector{Matrix{T}} 
    dP_dH::Vector{Matrix{T}} 
    dP_dA::Vector{Matrix{T}} 
    dP_dB::Vector{Matrix{T}} 

    dp_dQ::Vector{Matrix{T}} 
    dp_dR::Vector{Matrix{T}} 
    dp_dH::Vector{Matrix{T}} 
    dp_dq::Vector{Matrix{T}} 
    dp_dr::Vector{Matrix{T}} 
    dp_dA::Vector{Matrix{T}} 
    dp_dB::Vector{Matrix{T}} 
    dp_df::Vector{Matrix{T}} 

    # Recursive derivatives
    dK_dP::Vector{Matrix{T}} 
    dd_dP::Vector{Matrix{T}} 
    dd_dp::Vector{Matrix{T}} 

    dP_dP::Vector{Matrix{T}} 
    dp_dP::Vector{Matrix{T}} 
    dp_dp::Vector{Matrix{T}} 

    dP::Vector{Matrix{T}}
    dp::Vector{Matrix{T}}
end

function dLQR(n,m,N)
    Q = [diagm(rand(n)) for k = 1:N]
    R = [diagm(rand(m)) for k = 1:N-1]
    H = [zeros(m,n) for k = 1:N-1]
    q = [randn(n) for k = 1:N]
    r = [randn(m) for k = 1:N-1]
    A = [randn(n,n) for k = 1:N-1]
    B = [randn(n,m) for k = 1:N-1]
    f = [randn(n) for k = 1:N-1]

    Qxx = deepcopy(A)
    Qux = deepcopy(H)
    Quu = deepcopy(R)
    Qx = deepcopy(f)
    Qu = deepcopy(r)
    P = deepcopy(Q)
    p = deepcopy(q)
    K = deepcopy(H)
    d = deepcopy(r)

    dP_dQ = [zeros(n*n, n*n) for k = 1:N]
    dP_dR = [zeros(n*n, m*m) for k = 1:N-1]
    dP_dH = [zeros(n*n, m*n) for k = 1:N-1]
    dP_dA = [zeros(n*n, n*n) for k = 1:N-1]
    dP_dB = [zeros(n*n, n*m) for k = 1:N-1]

    dp_dQ = [zeros(n, n*n) for k = 1:N]
    dp_dR = [zeros(n, m*m) for k = 1:N-1]
    dp_dH = [zeros(n, m*n) for k = 1:N-1]
    dp_dq = [zeros(n, n) for k = 1:N]
    dp_dr = [zeros(n, m) for k = 1:N]
    dp_dA = [zeros(n, n*n) for k = 1:N-1]
    dp_dB = [zeros(n, n*m) for k = 1:N-1]
    dp_df = [zeros(n, n) for k = 1:N-1]

    # Recursive derivatives
    dK_dP = [zeros(m*n, n*n) for k = 1:N-1]
    dd_dP = [zeros(m, n*n) for k = 1:N-1]
    dd_dp = [zeros(m, n) for k = 1:N-1]

    dP_dP = [zeros(n*n, n*n) for k = 1:N-1]
    dp_dP = [zeros(n, n*n) for k = 1:N-1]
    dp_dp = [zeros(n, n) for k = 1:N-1]

    dP = deepcopy(dd_dP)
    dp = deepcopy(dd_dp)
    
    dLQR(
        Q, R, H, q, r, A, B, f,
        Qxx, Quu, Qux, Qx, Qu, P, p, K, d,
        dP_dQ, dP_dR, dP_dH, dP_dA, dP_dB,
        dp_dQ, dp_dR, dp_dH, dp_dq, dp_dr, dp_dA, dp_dB, dp_df,
        dK_dP, dd_dP, dd_dp,
        dP_dP, dp_dP, dp_dp,
        dP,dp
    )
end

function calc_derivatives!(prob::dLQR)
    p = prob
    dP_dQ,dP_dR,dP_dH = p.dP_dQ, p.dP_dR, p.dP_dH
    dp_dQ,dp_dR,dp_dH,dp_dq,dp_dr = p.dp_dQ, p.dp_dR, p.dp_dH, p.dp_dq, p.dp_dr
    dP_dA,dP_dB = p.dP_dA, p.dP_dB
    dp_dA,dp_dB,dp_df = p.dp_dA, p.dp_dB, p.dp_df
    dK_dP, dd_dP, dd_dp = p.dK_dP, p.dd_dP, p.dd_dp
    dP_dP, dp_dP, dp_dp = p.dP_dP, p.dp_dP, p.dp_dp
    dP,dp = p.dP, p.dp

    Q,R,H,q,r = p.Q, p.R, p.H, p.q, p.r
    A,B,f = p.A, p.B, p.f
    Qxx,Quu,Qux,Qx,Qu = p.Qxx,p.Quu,p.Qux,p.Qx,p.Qu
    K,d = p.K, p.d
    P,p = prob.P, prob.p

    n,m = size(B[1])
    N = length(Q)

    P[end] .= Q[end]
    p[end] .= q[end]
    dP_dQ[end] .= I(n*n)
    dp_dq[end] .= I(n)
    for k = reverse(1:N-1)
        Qxx[k] = Q[k] + A[k]'P[k+1]*A[k]
        Qux[k] = H[k] + B[k]'P[k+1]*A[k]
        Quu[k] = R[k] + B[k]'P[k+1]*B[k]
        Qx[k] = q[k] + A[k]'*(P[k+1] * f[k] + p[k+1])
        Qu[k] = r[k] + B[k]'*(P[k+1] * f[k] + p[k+1])

        K[k] .= Quu[k] \ Qux[k]
        d[k] .= Quu[k] \ Qu[k]

        let A=A[k], B=B[k], f=f[k], P=P[k+1], p=p[k+1], Quu=Quu[k], Qu=Qu[k], Qux=Qux[k], K=K[k], d=d[k]
            dQxx_dA = kron(A'P, I(n)) * Matrix(comm(n,n)) + kron(I(n), A'P)
            dQux_dA = kron(I(n), B'P)
            dQx_dA = kron(f'P + p', I(n)) * comm(n,n)

            dQuu_dB = kron(B'P, I(m)) * Matrix(comm(n,m)) + kron(I(m), B'P)
            dQux_dB = kron(A'P,I(m)) * comm(n,m)
            dQu_dB = kron(f'P + p', I(m)) * comm(n,m)

            dQxx_dP = kron(A',A')
            dQux_dP = kron(A',B')
            dQuu_dP = kron(B',B')

            dQx_dP = kron(f',A')
            dQu_dP = kron(f',B')

            dQx_dp = A'
            dQu_dp = B' 

            dK_dA = kron(I(n),inv(Quu))*dQux_dA
            dK_dB = kron(I(n),inv(Quu))*dQux_dB - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
            dK_dR = -kron(Qux', I(m))*kron(inv(Quu),inv(Quu))
            dK_dH = kron(I(n), inv(Quu))
            dK_dP[k] = kron(I(n),inv(Quu))*dQux_dP - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP

            dd_dB = inv(Quu)*dQu_dB - kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
            dd_dR = -kron(Qu', I(m))*kron(inv(Quu),inv(Quu))
            dd_dr = inv(Quu)
            dd_dP[k] = kron(f',Quu\B') - kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP
            dd_dp[k] = Quu\(B')

            # Derivatives wrt P
            dP_dK = kron(K'Quu, I(n))*comm(m,n) + kron(I(n), K'Quu) + kron(Qux',I(n))*comm(m,n) + kron(I(n),Qux')
            dP_dQuu = kron(K',K')
            dP_dQux = kron(I(n), K') + kron(K',I(n))*comm(m,n)
            dP_dA[k] = dQxx_dA + dP_dK * dK_dA + dP_dQux * dQux_dA
            dP_dB[k] = dP_dQuu * dQuu_dB + dP_dK * dK_dB + dP_dQux * dQux_dB
            dP_dQ[k] = Matrix(I,n*n,n*n) 
            dP_dR[k] = dP_dQuu + dP_dK * dK_dR
            dP_dH[k] = dP_dQux + dP_dK * dK_dH
            dP_dP[k] = dQxx_dP + dP_dQux * dQux_dP + dP_dQuu * dQuu_dP + dP_dK * dK_dP[k]


            # Derivatives wrt p
            dp_dK = kron(d'Quu + Qu', I(n))*comm(m,n)
            dp_dQuu = kron(d',K')
            dp_dQux = kron(d',I(n)) * comm(m,n) 
            dp_dQx = I(n) 
            dp_dQu = K' 
            dp_dd = K'Quu + Qux' 
            dp_dA[k] = dp_dQux * dQux_dA + dp_dQx * dQx_dA + dp_dK * dK_dA
            dp_dB[k] = dp_dQuu * dQuu_dB + dp_dQux * dQux_dB + dp_dQu * dQu_dB + dp_dK * dK_dB + dp_dd * dd_dB
            dp_df[k] = dp_dQx * A'P + dp_dQu * B'P + dp_dd * (Quu\(B'P))
            dp_dQ[k] = zeros(n,n*n) 
            dp_dR[k] = dp_dQuu + dp_dK * dK_dR + dp_dd * dd_dR
            dp_dH[k] = dp_dQux + dp_dK * dK_dH
            dp_dq[k] = dp_dQx
            dp_dr[k] = dp_dQu + dp_dd * dd_dr
            dp_dP[k] = dp_dQuu * dQuu_dP + dp_dQux * dQux_dP + dp_dQx * dQx_dP + dp_dQu * dQu_dP + dp_dK * dK_dP[k] + dp_dd * dd_dP[k]
            dp_dp[k] = dp_dQx * dQx_dp + dp_dQu * dQu_dp + dp_dd * dd_dp[k]
        end

        P[k] .= Qxx[k] + K[k]'Quu[k]*K[k] + K[k]'Qux[k] + Qux[k]'K[k]
        p[k] .= Qx[k] + K[k]'Quu[k]*d[k] + K[k]'Qu[k] + Qux[k]'d[k]
    end

    dP[1] .= dd_dP[1]
    dp[1] .= dd_dp[1]
    for k = 2:N-1
        dP[k] .= dP[k-1] * dP_dP[k] + dp[k-1] * dp_dP[k]
        dp[k] .= dp[k-1] * dp_dp[k]
    end
end

# function tvlqr(prob::dLQR)

#     T = promote_type(eltype(A[1]), eltype(B[1]), eltype(f[1]), eltype(Q[1]), eltype(R[1]), 
#         eltype(H[1]), eltype(q[1]), eltype(r[1])
#     )
#     n,m = size(B[1])
#     N = length(Q)
#     P = [zeros(T,n,n) for k = 1:N]
#     p = [zeros(T,n) for k = 1:N]
#     K = [zeros(T,m,n) for k = 1:N-1]
#     d = [zeros(T,m) for k = 1:N-1]
#     P[end] .= Q[end]
#     p[end] .= q[end]
#     for k = reverse(1:N-1)
#         Qxx = Q[k] + A[k]'P[k+1]*A[k]
#         Qux = H[k] + B[k]'P[k+1]*A[k]
#         Quu = R[k] + B[k]'P[k+1]*B[k]
#         Qx = q[k] + A[k]'*(P[k+1] * f[k] + p[k+1])
#         Qu = r[k] + B[k]'*(P[k+1] * f[k] + p[k+1])
        
#         K[k] .= -(Quu \ Qux)
#         d[k] .= -(Quu \ Qu)
#         P[k] .= Qxx + K[k]'Quu*K[k] + K[k]'Qux + Qux'K[k]
#         p[k] .= Qx + K[k]'Quu*d[k] + K[k]'Qu + Qux'd[k]
#     end
#     return K,d, P,p
# end

function (prob::dLQR)(x)
    # Extract out problem data
    p = prob
    Q,R,H,q,r = p.Q, p.R, p.H, p.q, p.r
    A,B,f = p.A, p.B, p.f

    # Evaluate TVLQR
    K,d, = tvlqr(A,B,f, Q,R,H,q,r)

    # Evaluate the first control
    K[1] * x + d[1]
end

function tvlqr(A,B,f,Q,R,H,q,r)
    T = promote_type(eltype(A[1]), eltype(B[1]), eltype(f[1]), eltype(Q[1]), eltype(R[1]), 
        eltype(H[1]), eltype(q[1]), eltype(r[1])
    )
    n,m = size(B[1])
    N = length(Q)
    P = [zeros(T,n,n) for k = 1:N]
    p = [zeros(T,n) for k = 1:N]
    K = [zeros(T,m,n) for k = 1:N-1]
    d = [zeros(T,m) for k = 1:N-1]
    P[end] .= Q[end]
    p[end] .= q[end]
    ΔV = zeros(T,2) 
    for k = reverse(1:N-1)
        Qxx = Q[k] + A[k]'P[k+1]*A[k]
        Qux = H[k] + B[k]'P[k+1]*A[k]
        Quu = R[k] + B[k]'P[k+1]*B[k]
        Qx = q[k] + A[k]'*(P[k+1] * f[k] + p[k+1])
        Qu = r[k] + B[k]'*(P[k+1] * f[k] + p[k+1])
        
        K[k] .= Quu \ Qux
        d[k] .= Quu \ Qu
        d[k] .*= -1
        P[k] .= Qxx + K[k]'Quu*K[k] - K[k]'Qux - Qux'K[k]
        p[k] .= Qx - K[k]'Quu*d[k] - K[k]'Qu + Qux'd[k]

        ΔV[1] += 0.5 * dot(d[k], Quu, d[k])
        ΔV[2] += dot(d[k], Qu)
    end
    return K,d, P,p,ΔV
end

function tvlqr(A,B,f, Q,R,H,q,r, x0)
    K,d,P,p = tvlqr(A,B,f, Q,R,H,q,r)
    m,n = size(K[1])
    Nmpc = length(Q)
    T = promote_type(eltype(K[1]), eltype(d[1]), eltype(x0))

    X = [zeros(T,n) for k = 1:Nmpc]
    U = [zeros(T,m) for k = 1:Nmpc-1]
    Y = [zeros(T,n) for k = 1:Nmpc]    

    X[1] .= x0
    for k = 1:Nmpc-1
        Y[k] = P[k] * X[k] + p[k]
        U[k] = -K[k] * X[k] + d[k]
        X[k+1] = A[k] * X[k] + B[k] * U[k] + f[k]
    end
    Y[Nmpc] = P[Nmpc] * X[Nmpc] + p[Nmpc]
    X,U,Y, K,d,P,p
end

function dlqr(A,B,f, Q,R,H,q,r)
    N = length(Q)
    n,m = size(B[1])
    T = promote_type(eltype(A[1]), eltype(B[1]), eltype(f[1]), eltype(Q[1]), eltype(R[1]), 
        eltype(H[1]), eltype(q[1]), eltype(r[1])
    )
    P = [zeros(T,n,n) for k = 1:N]
    p = [zeros(T,n) for k = 1:N]
    K = [zeros(T,m,n) for k = 1:N-1]
    d = [zeros(T,m) for k = 1:N-1]

    Qxx = [zeros(T,n,n) for k = 1:N-1]
    Quu = [zeros(T,m,m) for k = 1:N-1]
    Qux = [zeros(T,m,n) for k = 1:N-1]
    Qx = [zeros(T,n) for k = 1:N-1]
    Qu = [zeros(T,m) for k = 1:N-1]

    dP_dQ = [zeros(T,n*n, n*n) for k = 1:N]
    dP_dR = [zeros(T,n*n, m*m) for k = 1:N-1]
    dP_dH = [zeros(T,n*n, m*n) for k = 1:N-1]
    dP_dA = [zeros(T,n*n, n*n) for k = 1:N-1]
    dP_dB = [zeros(T,n*n, n*m) for k = 1:N-1]

    dp_dQ = [zeros(T,n, n*n) for k = 1:N]
    dp_dR = [zeros(T,n, m*m) for k = 1:N-1]
    dp_dH = [zeros(T,n, m*n) for k = 1:N-1]
    dp_dq = [zeros(T,n, n) for k = 1:N]
    dp_dr = [zeros(T,n, m) for k = 1:N]
    dp_dA = [zeros(T,n, n*n) for k = 1:N-1]
    dp_dB = [zeros(T,n, n*m) for k = 1:N-1]
    dp_df = [zeros(T,n, n) for k = 1:N-1]

    # Recursive derivatives
    dK_dP = [zeros(T,m*n, n*n) for k = 1:N-1]
    dd_dP = [zeros(T,m, n*n) for k = 1:N-1]
    dd_dp = [zeros(T,m, n) for k = 1:N-1]

    dP_dP = [zeros(T,n*n, n*n) for k = 1:N-1]
    dp_dP = [zeros(T,n, n*n) for k = 1:N-1]
    dp_dp = [zeros(T,n, n) for k = 1:N-1]

    dP_K = deepcopy(dK_dP)   # partial of K1 wrt P[1:N]
    dP = deepcopy(dd_dP)     # partial of d1 wrt P[1:N]
    dp = deepcopy(dd_dp)     # partial of d1 wrt p[1:N]
    
    dK_dA = zeros(m*n, n*n)
    dK_dB = zeros(m*n, n*m)
    dK_dQ = zeros(m*n, n*n)
    dK_dR = zeros(m*n, m*m)
    dK_dH = zeros(m*n, m*n)
    dd_dB = zeros(m, n*m)
    dd_dR = zeros(m, m*m)
    dd_dr = zeros(m, m)

    P[end] .= Q[end]
    p[end] .= q[end]
    dP_dQ[end] .= I(n*n)
    dp_dq[end] .= I(n)
    for k = reverse(1:N-1)
        Qxx[k] = Q[k] + A[k]'P[k+1]*A[k]
        Qux[k] = H[k] + B[k]'P[k+1]*A[k]
        Quu[k] = R[k] + B[k]'P[k+1]*B[k]
        Qx[k] = q[k] + A[k]'*(P[k+1] * f[k] + p[k+1])
        Qu[k] = r[k] + B[k]'*(P[k+1] * f[k] + p[k+1])

        K[k] .= Quu[k] \ Qux[k]
        d[k] .= Quu[k] \ Qu[k]
        d[k] .*= -1

        let A=A[k], B=B[k], f=f[k], P=P[k+1], p=p[k+1], Quu=Quu[k], Qu=Qu[k], Qux=Qux[k], K=K[k], d=d[k]
            dQxx_dA = kron(A'P, I(n)) * Matrix(comm(n,n)) + kron(I(n), A'P)
            dQux_dA = kron(I(n), B'P)
            dQx_dA = kron(f'P + p', I(n)) * comm(n,n)

            dQuu_dB = kron(B'P, I(m)) * Matrix(comm(n,m)) + kron(I(m), B'P)
            dQux_dB = kron(A'P,I(m)) * comm(n,m)
            dQu_dB = kron(f'P + p', I(m)) * comm(n,m)

            dQxx_dP = kron(A',A')
            dQux_dP = kron(A',B')
            dQuu_dP = kron(B',B')

            dQx_dP = kron(f',A')
            dQu_dP = kron(f',B')

            dQx_dp = A'
            dQu_dp = B' 

            dK_dA = kron(I(n),inv(Quu))*dQux_dA
            dK_dB = kron(I(n),inv(Quu))*dQux_dB - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
            dK_dR = -kron(Qux', I(m))*kron(inv(Quu),inv(Quu))
            dK_dH = kron(I(n), inv(Quu))
            dK_dP[k] = kron(I(n),inv(Quu))*dQux_dP - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP

            dd_dB .= -inv(Quu)*dQu_dB + kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
            dd_dR .= +kron(Qu', I(m))*kron(inv(Quu),inv(Quu))
            dd_dr .= -inv(Quu)
            dd_dP[k] = -kron(f',Quu\B') + kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP
            dd_dp[k] = -(Quu\(B'))

            # Derivatives wrt P
            dP_dK = kron(K'Quu, I(n))*comm(m,n) + kron(I(n), K'Quu) - kron(Qux',I(n))*comm(m,n) - kron(I(n),Qux')
            dP_dQuu = kron(K',K')
            dP_dQux = -kron(I(n), K') - kron(K',I(n))*comm(m,n)
            dP_dA[k] = dQxx_dA + dP_dK * dK_dA + dP_dQux * dQux_dA
            dP_dB[k] = dP_dQuu * dQuu_dB + dP_dK * dK_dB + dP_dQux * dQux_dB
            dP_dQ[k] = Matrix(I,n*n,n*n) 
            dP_dR[k] = dP_dQuu + dP_dK * dK_dR
            dP_dH[k] = dP_dQux + dP_dK * dK_dH
            dP_dP[k] = dQxx_dP + dP_dQux * dQux_dP + dP_dQuu * dQuu_dP + dP_dK * dK_dP[k]


            # Derivatives wrt p
            dp_dK = -kron(d'Quu + Qu', I(n))*comm(m,n)
            dp_dQuu = -kron(d',K')
            dp_dQux = kron(d',I(n)) * comm(m,n) 
            dp_dQx = I(n) 
            dp_dQu = -K' 
            dp_dd = -K'Quu + Qux' 
            dp_dA[k] = dp_dQux * dQux_dA + dp_dQx * dQx_dA + dp_dK * dK_dA
            dp_dB[k] = dp_dQuu * dQuu_dB + dp_dQux * dQux_dB + dp_dQu * dQu_dB + dp_dK * dK_dB + dp_dd * dd_dB
            dp_df[k] = dp_dQx * A'P + dp_dQu * B'P + dp_dd * (Quu\(B'P))
            dp_dQ[k] = zeros(n,n*n) 
            dp_dR[k] = dp_dQuu + dp_dK * dK_dR + dp_dd * dd_dR
            dp_dH[k] = dp_dQux + dp_dK * dK_dH
            dp_dq[k] = dp_dQx
            dp_dr[k] = dp_dQu + dp_dd * dd_dr
            dp_dP[k] = dp_dQuu * dQuu_dP + dp_dQux * dQux_dP + dp_dQx * dQx_dP + dp_dQu * dQu_dP + dp_dK * dK_dP[k] + dp_dd * dd_dP[k]
            dp_dp[k] = dp_dQx * dQx_dp + dp_dQu * dQu_dp + dp_dd * dd_dp[k]
        end

        P[k] .= Qxx[k] + K[k]'Quu[k]*K[k] - K[k]'Qux[k] - Qux[k]'K[k]
        p[k] .= Qx[k] - K[k]'Quu[k]*d[k] - K[k]'Qu[k] + Qux[k]'d[k]
    end

    dP[1] .= dd_dP[1]
    dp[1] .= dd_dp[1]
    dP_K[1] .= dK_dP[1]
    for k = 2:N-1
        dP[k] .= dP[k-1] * dP_dP[k] + dp[k-1] * dp_dP[k]
        dp[k] .= dp[k-1] * dp_dp[k]
        dP_K[k] .= dP_K[k-1] * dP_dP[k]
    end
    return (;
        K, d, P, p,
        dP, dp, dP_K, 
        dP_dA, dP_dB, dP_dQ, dP_dR, dP_dH, 
        dp_dA, dp_dB, dp_df, dp_dQ, dp_dR, dp_dH, dp_dq, dp_dr,
        dd_dB, dd_dR, dd_dr, dK_dA, dK_dB, dK_dQ, dK_dR, dK_dH,
    )
end

function tvlqr_closed_loop_derivatives(dyn,jac,x0,theta,h,build_tvlqr, n,m,Nsim,Nmpc; 
        dA=nothing,dB=nothing,df=nothing,dQ=nothing,dR=nothing,dH=nothing,dq=nothing,dr=nothing)

    params = filter(!isnothing, [dA,dB,df,dQ,dR,dH,dq,dr])
    @assert length(params) > 0
    p = length(theta) 
    @assert all(param->size(param[1],2)==p, params)

    X = [zeros(n) for k = 1:Nsim]
    U = [zeros(m) for k = 1:Nsim-1]
    dX = [zeros(n,n+m) for k = 1:Nsim]
    dU = [zeros(m,n+m) for k = 1:Nsim-1]

    X[1] .= x0
    dX[1] .= 0
    for i = 1:Nsim-1

        # Build the TVLQR problem about the reference trajectory
        # This is problem-dependent, since the input in an arbitrary theta
        # For ALTRO, we'll need to pass in the initial trajectory as well
        tvlqr_data = build_tvlqr(theta,Nmpc,i)
        xref = tvlqr_data[1]
        uref = tvlqr_data[2]
        ∂ = dlqr(tvlqr_data[3:end]...)       # calculate derivatives of TVLQR problem

        # Simulate the system forward
        dx = X[i] - xref[1]           # state error at first time step
        du = -∂.K[1] * dx + ∂.d[1]    # get control
        U[i] = uref[1] + du
        X[i+1] = dyn(X[i], U[i], h)   # simulate forward

        # Get the Jacobian of the simulated trajectory
        Ji = jac(X[i], U[i], h)
        Ai = Ji[:,1:n]
        Bi = Ji[:,n+1:end]
    
        # Calculate derivatives wrt parameters
        # Since the parameters apply at each time step, sum the derivatives of each time step
        dd_dθ = mapreduce(+,1:Nmpc) do k
            dd = zeros(m,p)
            if !isnothing(dQ) && k > 1
                dd_dQk = ∂.dP[k-1] * ∂.dP_dQ[k] + ∂.dp[k-1] * ∂.dp_dQ[k]
                dd += dd_dQk * dQ[k]
            end
            if !isnothing(dR) && k < Nmpc
                dd_dRk = if k > 1 
                    ∂.dP[k-1] * ∂.dP_dR[k] + ∂.dp[k-1] * ∂.dp_dR[k]
                else
                    ∂.dd_dR
                end
                dd += dd_dRk * dR[k]
            end
            dd
        end
        dK_dθ = mapreduce(+,1:Nmpc) do k
            dK = zeros(m*n,p)
            if !isnothing(dQ) && k > 1
                dK_dQk = ∂.dP_K[k-1] * ∂.dP_dQ[k]
                dK += dK_dQk * dQ[k]
            end
            if !isnothing(dR) && k < Nmpc
                dK_dRk = if k > 1
                    ∂.dP_K[k-1] * ∂.dP_dR[k]
                else
                    ∂.dK_dR
                end
                dK += dK_dRk * dR[k]
            end
            dK
        end
    
        # Calculate total derivative of u1 wrt the parameters θ
        dU[i] = -kron(dx',I(m)) * dK_dθ + dd_dθ - ∂.K[1] * dX[i]
    
        # Propagate the derivatives to the next time step through the simulated dynamics 
        dX[i+1] = Ai * dX[i] + Bi * dU[i]
    end
    X,U, dX,dU
end