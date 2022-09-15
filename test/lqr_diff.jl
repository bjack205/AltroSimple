using LinearAlgebra
using ForwardDiff
using MatrixCalculus

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
    for k = reverse(1:N-1)
        Qxx = Q[k] + A[k]'P[k+1]*A[k]
        Qux = H[k] + B[k]'P[k+1]*A[k]
        Quu = R[k] + B[k]'P[k+1]*B[k]
        Qx = q[k] + A[k]'*(P[k+1] * f[k] + p[k+1])
        Qu = r[k] + B[k]'*(P[k+1] * f[k] + p[k+1])
        
        K[k] .= Quu \ Qux
        d[k] .= Quu \ Qu
        P[k] .= Qxx + K[k]'Quu*K[k] + K[k]'Qux + Qux'K[k]
        p[k] .= Qx + K[k]'Quu*d[k] + K[k]'Qu + Qux'd[k]
    end
    return K,d, P,p
end

const ⊗ = kron

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
dQxx_dA ≈ ForwardDiff.jacobian(A->calcQxx(Q,A,P), A)

dQux_dA = kron(I(n), B'P)
dQux_dA ≈ ForwardDiff.jacobian(A->calcQux(H,A,B,P), A)

dQx_dA = kron(f'P + p', I(n)) * comm(n,n)
dQx_dA ≈ ForwardDiff.jacobian(A->calcQx(q,A,f,P,p), A)

dQuu_dB = kron(B'P, I(m)) * Matrix(comm(n,m)) + kron(I(m), B'P)
dQuu_dB ≈ ForwardDiff.jacobian(B->calcQuu(R,B,P), B)

dQux_dB = kron(A'P,I(m)) * comm(n,m)
dQux_dB ≈ ForwardDiff.jacobian(B->calcQux(H,A,B,P), B)

dQu_dB = kron(f'P + p', I(m)) * comm(n,m)
dQu_dB ≈ ForwardDiff.jacobian(B->calcQu(r,B,f,P,p), B)

dQxx_dP = kron(A',A')
dQxx_dP ≈ ForwardDiff.jacobian(P->calcQxx(Q,A,P), P)

dQux_dP = kron(A',B')
dQux_dP ≈ ForwardDiff.jacobian(P->calcQux(H,A,B,P), P)

dQuu_dP = kron(B',B')
dQuu_dP ≈ ForwardDiff.jacobian(P->calcQuu(R,B,P), P)

dQx_dP = kron(f',A')
dQx_dP ≈ ForwardDiff.jacobian(P->calcQx(q,A,f,P,p), P)

dQu_dP = kron(f',B')
dQu_dP ≈ ForwardDiff.jacobian(P->calcQu(r,B,f,P,p), P)

dQx_dp = A'
dQx_dp ≈ ForwardDiff.jacobian(p->calcQx(q,A,f,P,p), p)

dQu_dp = B' 
dQu_dp ≈ ForwardDiff.jacobian(p->calcQu(r,B,f,P,p), p)

##
calcK(A,B,R,H,P) = (R + B'P*B)\(H + B'P*A)
calcd(B,f,R,r,P,p) = (R + B'P*B)\(r + B'*(P*f + p))

Qxx = calcQxx(Q,A,P)
Qux = calcQux(H,A,B,P)
Quu = calcQuu(R,B,P)
Qx = calcQx(q,A,f,P,p)
Qu = calcQu(r,B,f,P,p)

calcK(A,B,R,H,P) ≈ Quu\Qux

dK_dA = kron(I(n),inv(Quu))*dQux_dA
dK_dA ≈ ForwardDiff.jacobian(A->calcK(A,B,R,H,P),A)

dK_dB = kron(I(n),inv(Quu))*dQux_dB - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
dK_dB ≈ ForwardDiff.jacobian(B->calcK(A,B,R,H,P),B)

dK_dR = -kron(Qux', I(m))*kron(inv(Quu),inv(Quu))
dK_dR ≈ ForwardDiff.jacobian(R->calcK(A,B,R,H,P),R)

dK_dH = kron(I(n), inv(Quu))
dK_dH ≈ ForwardDiff.jacobian(H->calcK(A,B,R,H,P),H)

dK_dP = kron(I(n),inv(Quu))*dQux_dP - kron(Qux', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP
dK_dP ≈ ForwardDiff.jacobian(P->calcK(A,B,R,H,P),P)

dd_dB = inv(Quu)*dQu_dB - kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dB
dd_dB ≈ ForwardDiff.jacobian(B->calcd(B,f,R,r,P,p),B)

dd_dR = -kron(Qu', I(m))*kron(inv(Quu),inv(Quu))
dd_dR ≈ ForwardDiff.jacobian(R->calcd(B,f,R,r,P,p),R)

dd_dP = kron(f',Quu\B') - kron(Qu', I(m))*kron(inv(Quu),inv(Quu))*dQuu_dP
dd_dP ≈ ForwardDiff.jacobian(P->calcd(B,f,R,r,P,p),P)

##
calcP(Qxx,Quu,Qux,K) = Qxx + K'Quu*K + K'Qux + Qux'K
calcP(A,B,Q,R,H,P) = begin
    Qxx = calcQxx(Q, A, P)
    Qux = calcQux(H, A, B, P)
    Quu = calcQuu(R, B, P)
    K = calcK(A,B, R, H, P)
    calcP(Qxx,Quu,Qux,K)
end
calcp(Quu,Qux,Qx,Qu,K,d) = Qx + K'Quu*d + K'Qu + Qux'd
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

dP_dK = kron(K'Quu, I(n))*comm(m,n) + kron(I(n), K'Quu) + kron(Qux',I(n))*comm(m,n) + kron(I(n),Qux')
dP_dK ≈ ForwardDiff.jacobian(K->calcP(Qxx,Quu,Qux,K),K)

dP_dQuu = kron(K',K')
dP_dQuu ≈ ForwardDiff.jacobian(Quu->calcP(Qxx,Quu,Qux,K),Quu)

dP_dQux = kron(I(n), K') + kron(K',I(n))*comm(m,n)
dP_dQux ≈ ForwardDiff.jacobian(Qux->calcP(Qxx,Quu,Qux,K),Qux)

dP_dA = dQxx_dA + dP_dK * dK_dA + dP_dQux * dQux_dA
dP_dA ≈ ForwardDiff.jacobian(A->calcP(A,B,Q,R,H,P),A)

dP_dB = dP_dQuu * dQuu_dB + dP_dK * dK_dB + dP_dQux * dQux_dB
dP_dB ≈ ForwardDiff.jacobian(B->calcP(A,B,Q,R,H,P),B)

dP_dQ = Matrix(I,n*n,n*n) 
dP_dQ ≈ ForwardDiff.jacobian(Q->calcP(A,B,Q,R,H,P),Q)

dP_dR = dP_dQuu + dP_dK * dK_dR
dP_dR ≈ ForwardDiff.jacobian(R->calcP(A,B,Q,R,H,P),R)

dP_dH = dP_dQux + dP_dK * dK_dH
dP_dH ≈ ForwardDiff.jacobian(H->calcP(A,B,Q,R,H,P),H)

dP_dP = dQxx_dP + dP_dQux * dQux_dP + dP_dQuu * dQuu_dP + dP_dK * dK_dP
dP_dP ≈ ForwardDiff.jacobian(P->calcP(A,B,Q,R,H,P),P)

dp_dP = dQxx_dP + dP_dQux * dQux_dP + dP_dQuu * dQuu_dP + dP_dK * dK_dP

##
A = [randn(n,n) for k = 1:N-1]
B = [randn(n,m) for k = 1:N-1]
f = [randn(n) for k = 1:N-1]
Q = [diagm(rand(n)) for k = 1:N]
R = [diagm(rand(m)) for k = 1:N-1]
H = [zeros(m,n) for k = 1:N]
q = [randn(n) for k = 1:N]
r = [randn(m) for k = 1:N-1]

K,d,P,p = tvlqr(A,B,f, Q,R,H,q,r)
ForwardDiff.jacobian(Kk) do
    K_ = similar.(K,eltype(Kk))
end
@run tvlqr(A,B,f, Q,R,H,q,r)