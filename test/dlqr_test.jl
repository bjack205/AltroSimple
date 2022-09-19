using SimpleAltro
using ForwardDiff
using Test

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

n,m,N = 3,2,5
prob = SimpleAltro.dLQR(n,m,N)
SimpleAltro.calc_derivatives!(prob)

p = prob
Q,R,H,q,r = p.Q, p.R, p.H, p.q, p.r
A,B,f = p.A, p.B, p.f

k = N-1
dx = zeros(n)
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

@test du1_dAk ≈ p.dP[k-1] * p.dP_dA[k] + p.dp[k-1] * p.dp_dA[k]
@test du1_dBk ≈ p.dP[k-1] * p.dP_dB[k] + p.dp[k-1] * p.dp_dB[k]
@test du1_dQk ≈ p.dP[k-1] * p.dP_dQ[k] + p.dp[k-1] * p.dp_dQ[k]
@test du1_dRk ≈ p.dP[k-1] * p.dP_dR[k] + p.dp[k-1] * p.dp_dR[k]
@test du1_dHk ≈ p.dP[k-1] * p.dP_dH[k] + p.dp[k-1] * p.dp_dH[k]
@test du1_dfk ≈ p.dp[k-1] * p.dp_df[k]
@test du1_dqk ≈ p.dp[k-1] * p.dp_dq[k]
@test du1_drk ≈ p.dp[k-1] * p.dp_dr[k]