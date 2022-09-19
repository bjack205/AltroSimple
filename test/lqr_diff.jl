using LinearAlgebra
using ForwardDiff
using MatrixCalculus
using SimpleAltro: tvlqr, dlqr
using Test

# function tvlqr(A,B,f,Q,R,H,q,r)
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
        
#         K[k] .= Quu \ Qux
#         d[k] .= Quu \ Qu
#         P[k] .= Qxx + K[k]'Quu*K[k] + K[k]'Qux + Qux'K[k]
#         p[k] .= Qx + K[k]'Quu*d[k] + K[k]'Qu + Qux'd[k]
#     end
#     return K,d, P,p
# end

# const ⊗ = kron

##
n,m,N = 3,2,5
A = randn(n,n)
B = randn(n,m)
f = randn(n)
Q = diagm(rand(n))
R = diagm(rand(m))
H = randn(m,n)
q = randn(n)
r = randn(m)
P = randn(n,n); P = P'P
p = randn(n)

## Test action-value expansion derivatives
calcQxx(Q, A, P) = Q + A'P*A
calcQux(H, A, B, P) = H + B'P*A
calcQuu(R, B, P) = R + B'P*B
calcQx(q, A, f, P, p) = q + A'*(P * f + p)
calcQu(r, B, f, P, p) = r + B'*(P * f + p)
dQxx_dA = kron(A'P, I(n)) * Matrix(comm(n,n)) + kron(I(n), A'P)
@test dQxx_dA ≈ ForwardDiff.jacobian(A->calcQxx(Q,A,P), A)

dQux_dA = kron(I(n), B'P)
@test dQux_dA ≈ ForwardDiff.jacobian(A->calcQux(H,A,B,P), A)

dQx_dA = kron(f'P + p', I(n)) * comm(n,n)
@test dQx_dA ≈ ForwardDiff.jacobian(A->calcQx(q,A,f,P,p), A)

dQuu_dB = kron(B'P, I(m)) * Matrix(comm(n,m)) + kron(I(m), B'P)
@test dQuu_dB ≈ ForwardDiff.jacobian(B->calcQuu(R,B,P), B)

dQux_dB = kron(A'P,I(m)) * comm(n,m)
@test dQux_dB ≈ ForwardDiff.jacobian(B->calcQux(H,A,B,P), B)

dQu_dB = kron(f'P + p', I(m)) * comm(n,m)
@test dQu_dB ≈ ForwardDiff.jacobian(B->calcQu(r,B,f,P,p), B)

dQxx_dP = kron(A',A')
@test dQxx_dP ≈ ForwardDiff.jacobian(P->calcQxx(Q,A,P), P)

dQux_dP = kron(A',B')
@test dQux_dP ≈ ForwardDiff.jacobian(P->calcQux(H,A,B,P), P)

dQuu_dP = kron(B',B')
@test dQuu_dP ≈ ForwardDiff.jacobian(P->calcQuu(R,B,P), P)

dQx_dP = kron(f',A')
@test dQx_dP ≈ ForwardDiff.jacobian(P->calcQx(q,A,f,P,p), P)

dQu_dP = kron(f',B')
@test dQu_dP ≈ ForwardDiff.jacobian(P->calcQu(r,B,f,P,p), P)

dQx_dp = A'
@test dQx_dp ≈ ForwardDiff.jacobian(p->calcQx(q,A,f,P,p), p)

dQu_dp = B' 
@test dQu_dp ≈ ForwardDiff.jacobian(p->calcQu(r,B,f,P,p), p)

##
calcK(A,B,R,H,P) = (R + B'P*B)\(H + B'P*A)
calcd(B,f,R,r,P,p) = -(R + B'P*B)\(r + B'*(P*f + p))

Qxx = calcQxx(Q,A,P)
Qux = calcQux(H,A,B,P)
Quu = calcQuu(R,B,P)
Qx = calcQx(q,A,f,P,p)
Qu = calcQu(r,B,f,P,p)

calcK(A,B,R,H,P) ≈ Quu\Qux

dK_dA = kron(I(n),inv(Quu))*dQux_dA
@test dK_dA ≈ ForwardDiff.jacobian(A->calcK(A,B,R,H,P),A)

dK_dB = kron(I(n),inv(Quu))*dQux_dB - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
@test dK_dB ≈ ForwardDiff.jacobian(B->calcK(A,B,R,H,P),B)

dK_dR = -kron(Qux', I(m))*kron(inv(Quu),inv(Quu))
@test dK_dR ≈ ForwardDiff.jacobian(R->calcK(A,B,R,H,P),R)

dK_dH = kron(I(n), inv(Quu))
@test dK_dH ≈ ForwardDiff.jacobian(H->calcK(A,B,R,H,P),H)

dK_dP = kron(I(n),inv(Quu))*dQux_dP - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP
@test dK_dP ≈ ForwardDiff.jacobian(P->calcK(A,B,R,H,P),P)

dd_dB = -inv(Quu)*dQu_dB + kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
@test dd_dB ≈ ForwardDiff.jacobian(B->calcd(B,f,R,r,P,p),B)

dd_dR = +kron(Qu', I(m))*kron(inv(Quu),inv(Quu))
@test dd_dR ≈ ForwardDiff.jacobian(R->calcd(B,f,R,r,P,p),R)

dd_dr = -inv(Quu)
@test dd_dr ≈ ForwardDiff.jacobian(r->calcd(B,f,R,r,P,p),r)

dd_dP = -kron(f',Quu\B') + kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP
@test dd_dP ≈ ForwardDiff.jacobian(P->calcd(B,f,R,r,P,p),P)

dd_dp = -(Quu\(B'))
@test dd_dp ≈ ForwardDiff.jacobian(p->calcd(B,f,R,r,P,p),p)

##
calcP(Qxx,Quu,Qux,K) = Qxx + K'Quu*K - K'Qux - Qux'K
calcP(A,B,Q,R,H,P) = begin
    Qxx = calcQxx(Q, A, P)
    Qux = calcQux(H, A, B, P)
    Quu = calcQuu(R, B, P)
    K = calcK(A,B, R, H, P)
    calcP(Qxx,Quu,Qux,K)
end
calcp(Quu,Qux,Qx,Qu,K,d) = Qx - K'Quu*d - K'Qu + Qux'd
calcp(A,B,f,Q,R,H,q,r,P,p) = begin
    Qux = calcQux(H, A, B, P)
    Quu = calcQuu(R, B, P)
    Qx = calcQx(q, A,f, P,p)
    Qu = calcQu(r, B,f, P,p)
    K = calcK(A,B, R, H, P)
    d = calcd(B,f, R,r, P,p)
    Qx + K'Quu*d + K'Qu + Qux'd
end

K = calcK(A,B,R,H,P)
d = calcd(B,f, R,r, P,p)

dP_dK = kron(K'Quu, I(n))*comm(m,n) + kron(I(n), K'Quu) - kron(Qux',I(n))*comm(m,n) - kron(I(n),Qux')
@test dP_dK ≈ ForwardDiff.jacobian(K->calcP(Qxx,Quu,Qux,K),K) atol=1e-12

dP_dQuu = kron(K',K')
@test dP_dQuu ≈ ForwardDiff.jacobian(Quu->calcP(Qxx,Quu,Qux,K),Quu)

dP_dQux = -kron(I(n), K') - kron(K',I(n))*comm(m,n)
@test dP_dQux ≈ ForwardDiff.jacobian(Qux->calcP(Qxx,Quu,Qux,K),Qux)

dP_dA = dQxx_dA + dP_dK * dK_dA + dP_dQux * dQux_dA
@test dP_dA ≈ ForwardDiff.jacobian(A->calcP(A,B,Q,R,H,P),A)

dP_dB = dP_dQuu * dQuu_dB + dP_dK * dK_dB + dP_dQux * dQux_dB
@test dP_dB ≈ ForwardDiff.jacobian(B->calcP(A,B,Q,R,H,P),B)

dP_dQ = Matrix(I,n*n,n*n) 
@test dP_dQ ≈ ForwardDiff.jacobian(Q->calcP(A,B,Q,R,H,P),Q)

dP_dR = dP_dQuu + dP_dK * dK_dR
@test dP_dR ≈ ForwardDiff.jacobian(R->calcP(A,B,Q,R,H,P),R)

dP_dH = dP_dQux + dP_dK * dK_dH
@test dP_dH ≈ ForwardDiff.jacobian(H->calcP(A,B,Q,R,H,P),H)

dP_dP = dQxx_dP + dP_dQux * dQux_dP + dP_dQuu * dQuu_dP + dP_dK * dK_dP
@test dP_dP ≈ ForwardDiff.jacobian(P->calcP(A,B,Q,R,H,P),P)

dp_dK = -kron(d'Quu + Qu', I(n))*comm(m,n)
@test dp_dK ≈ ForwardDiff.jacobian(K->calcp(Quu,Qux,Qx,Qu,K,d),K) atol=1e-12

dp_dQuu = -kron(d',K')
@test dp_dQuu ≈ ForwardDiff.jacobian(Quu->calcp(Quu,Qux,Qx,Qu,K,d),Quu)

dp_dQux = kron(d',I(n)) * comm(m,n) 
@test dp_dQux ≈ ForwardDiff.jacobian(Qux->calcp(Quu,Qux,Qx,Qu,K,d),Qux)

dp_dQx = I(n) 
@test dp_dQx ≈ ForwardDiff.jacobian(Qx->calcp(Quu,Qux,Qx,Qu,K,d),Qx)

dp_dQu = -K' 
@test dp_dQu ≈ ForwardDiff.jacobian(Qu->calcp(Quu,Qux,Qx,Qu,K,d),Qu)

dp_dd = -K'Quu + Qux' 
@test dp_dd ≈ ForwardDiff.jacobian(d->calcp(Quu,Qux,Qx,Qu,K,d),d)

dp_dA = dp_dQux * dQux_dA + dp_dQx * dQx_dA + dp_dK * dK_dA
@test dp_dA ≈ ForwardDiff.jacobian(A->calcp(A,B,f,Q,R,H,q,r,P,p),A)

dp_dB = dp_dQuu * dQuu_dB + dp_dQux * dQux_dB + dp_dQu * dQu_dB + dp_dK * dK_dB + dp_dd * dd_dB
@test dp_dB ≈ ForwardDiff.jacobian(B->calcp(A,B,f,Q,R,H,q,r,P,p),B)

dp_df = dp_dQx * A'P + dp_dQu * B'P + dp_dd * (Quu\(B'P))
@test dp_df ≈ ForwardDiff.jacobian(f->calcp(A,B,f,Q,R,H,q,r,P,p),f)

dp_dQ = zeros(n,n*n) 
@test dp_dQ ≈ ForwardDiff.jacobian(Q->calcp(A,B,f,Q,R,H,q,r,P,p),Q)

dp_dR = dp_dQuu + dp_dK * dK_dR + dp_dd * dd_dR
@test dp_dR ≈ ForwardDiff.jacobian(R->calcp(A,B,f,Q,R,H,q,r,P,p),R)

dp_dH = dp_dQux + dp_dK * dK_dH
@test dp_dH ≈ ForwardDiff.jacobian(H->calcp(A,B,f,Q,R,H,q,r,P,p),H)

dp_dq = dp_dQx
@test dp_dq ≈ ForwardDiff.jacobian(q->calcp(A,B,f,Q,R,H,q,r,P,p),q)

dp_dr = dp_dQu + dp_dd * dd_dr
@test dp_dr ≈ ForwardDiff.jacobian(r->calcp(A,B,f,Q,R,H,q,r,P,p),r)

dp_dP = dp_dQuu * dQuu_dP + dp_dQux * dQux_dP + dp_dQx * dQx_dP + dp_dQu * dQu_dP + dp_dK * dK_dP + dp_dd * dd_dP 
@test dp_dP ≈ ForwardDiff.jacobian(P->calcp(A,B,f,Q,R,H,q,r,P,p),P)

dp_dp = dp_dQx * dQx_dp + dp_dQu * dQu_dp + dp_dd * dd_dp
@test dp_dp ≈ ForwardDiff.jacobian(p->calcp(A,B,f,Q,R,H,q,r,P,p),p)


##
N = 6
A = [randn(n,n) for k = 1:N-1]
B = [randn(n,m) for k = 1:N-1]
f = [randn(n) for k = 1:N-1]
Q = [diagm(rand(n)) for k = 1:N]
R = [diagm(rand(m)) for k = 1:N-1]
H = [zeros(m,n) for k = 1:N-1]
q = [randn(n) for k = 1:N]
r = [randn(m) for k = 1:N-1]

P = deepcopy(Q)
p = deepcopy(q)
Qxx = deepcopy(A)
Qux = deepcopy(H)
Quu = deepcopy(R)
Qx = deepcopy(f)
Qu = deepcopy(r)

K = deepcopy(H)
d = deepcopy(r)

# Derivatives of parameters wrt cost-to-go
dP_dA = [zeros(n*n, n*n) for k = 1:N-1]
dP_dB = [zeros(n*n, n*m) for k = 1:N-1]
dP_dQ = [zeros(n*n, n*n) for k = 1:N]
dP_dR = [zeros(n*n, m*m) for k = 1:N-1]
dP_dH = [zeros(n*n, m*n) for k = 1:N-1]

dp_dA = [zeros(n, n*n) for k = 1:N-1]
dp_dB = [zeros(n, n*m) for k = 1:N-1]
dp_df = [zeros(n, n) for k = 1:N-1]
dp_dQ = [zeros(n, n*n) for k = 1:N]
dp_dR = [zeros(n, m*m) for k = 1:N-1]
dp_dH = [zeros(n, m*n) for k = 1:N-1]
dp_dq = [zeros(n, n) for k = 1:N]
dp_dr = [zeros(n, m) for k = 1:N]

# Recursive derivatives
dK_dP = [zeros(m*n, n*n) for k = 1:N-1]
dd_dP = [zeros(m, n*n) for k = 1:N-1]
dd_dp = [zeros(m, n) for k = 1:N-1]

dP_dP = [zeros(n*n, n*n) for k = 1:N-1]
dp_dP = [zeros(n, n*n) for k = 1:N-1]
dp_dp = [zeros(n, n) for k = 1:N-1]

dd_dB = zeros(m,m*n)
dd_dR = zeros(m,m*m)
dd_dr = zeros(m,m)

# Calculate stage-wise derivatives
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

        dd_dB = -inv(Quu)*dQu_dB + kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
        dd_dR = +kron(Qu', I(m))*kron(inv(Quu),inv(Quu))
        dd_dr = -inv(Quu)
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


dP = deepcopy(dd_dP)
dp = deepcopy(dd_dp)
dP[1] .= dd_dP[1]
dp[1] .= dd_dp[1]
for k = 2:N-1
    dP[k] .= dP[k-1] * dP_dP[k] + dp[k-1] * dp_dP[k]
    dp[k] .= dp[k-1] * dp_dp[k]
end

dP_,dp_ = dlqr(A,B,f, Q,R,H,q,r)
dP_ ≈ dP
dp_ ≈ dp

##
dx = zeros(n)
k = 4
du1_dAk = ForwardDiff.jacobian(A[k]) do Ak
    A_ = similar.(A,eltype(Ak))
    for i = 1:N-1
        A_[i] .= A[i]
    end
    A_[k] .= Ak
    K_,d_ = tvlqr(A_,B,f, Q,R,H, q,r)
    K_[1] * dx + d_[1]
end
du1_dBk = ForwardDiff.jacobian(B[k]) do Bk
    B_ = similar.(B,eltype(Bk))
    for i = 1:N-1
        B_[i] .= B[i]
    end
    B_[k] .= Bk
    K_,d_ = tvlqr(A,B_,f, Q,R,H, q,r)
    K_[1] * dx + d_[1]
end
du1_dQk = ForwardDiff.jacobian(Q[k]) do Qk
    Q_ = similar.(Q,eltype(Qk))
    for i = 1:N
        Q_[i] .= Q[i]
    end
    Q_[k] .= Qk
    K_,d_ = tvlqr(A,B,f, Q_,R,H, q,r)
    K_[1] * dx + d_[1]
end
du1_dRk = ForwardDiff.jacobian(R[k]) do Rk
    R_ = similar.(R,eltype(Rk))
    for i = 1:N-1
        R_[i] .= R[i]
    end
    R_[k] .= Rk
    K_,d_ = tvlqr(A,B,f, Q,R_,H, q,r)
    K_[1] * dx + d_[1]
end
du1_dHk = ForwardDiff.jacobian(H[k]) do Hk
    H_ = similar.(H,eltype(Hk))
    for i = 1:N-1
        H_[i] .= H[i]
    end
    H_[k] .= Hk
    K_,d_ = tvlqr(A,B,f, Q,R,H_, q,r)
    K_[1] * dx + d_[1]
end
du1_dfk = ForwardDiff.jacobian(f[k]) do fk
    f_ = similar.(f,eltype(fk))
    for i = 1:N-1
        f_[i] .= f[i]
    end
    f_[k] .= fk
    K_,d_ = tvlqr(A,B,f_, Q,R,H, q,r)
    K_[1] * dx + d_[1]
end
du1_dqk = ForwardDiff.jacobian(q[k]) do qk
    q_ = similar.(q,eltype(qk))
    for i = 1:N
        q_[i] .= q[i]
    end
    q_[k] .= qk
    K_,d_ = tvlqr(A,B,f, Q,R,H, q_,r)
    K_[1] * dx + d_[1]
end
du1_drk = ForwardDiff.jacobian(r[k]) do rk
    r_ = similar.(r,eltype(rk))
    for i = 1:N-1
        r_[i] .= r[i]
    end
    r_[k] .= rk
    K_,d_ = tvlqr(A,B,f, Q,R,H, q,r_)
    K_[1] * dx + d_[1]
end
du1_dAk
du1_dBk
du1_dfk
du1_dQk
du1_dRk
du1_dHk
du1_dqk
du1_drk

dP[k-1] * dP_dA[k] + dp[k-1] * dp_dA[k]
dP[k-1] * dP_dB[k] + dp[k-1] * dp_dB[k]
dp[k-1] * dp_df[k]
dP[k-1] * dP_dQ[k] + dp[k-1] * dp_dQ[k]
dP[k-1] * dP_dR[k] + dp[k-1] * dp_dR[k]
dP[k-1] * dP_dH[k] + dp[k-1] * dp_dH[k]
dp[k-1] * dp_dq[k]
dp[k-1] * dp_dr[k]
