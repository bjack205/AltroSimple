using LinearAlgebra
import SimpleAltro: tvlqr, dlqr 
using RobotZoo
using FiniteDiff
using ForwardDiff
import RobotDynamics as RD
using Plots
using Printf
using Test

include("recipes.jl")
include("scotty.jl")

evec(n,i) = begin e = zeros(n); e[i] = 1; e end

struct TrackingObjective{T}
    f::Function     # dynamics function
    df::Function    # dynamics Jacobian

    # Reference trajectory
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}

    # MPC params
    Nmpc::Int
end

function build_tvlqr(fd::Function, df::Function, Xref, Uref, Qd, Rd)
    T = promote_type(eltype(Qd), eltype(Rd))
    n = length(Xref[1])
    m = length(Uref[1])
    Nmpc = length(Xref)
    A = [zeros(T,n,n) for k = 1:Nmpc-1]
    B = [zeros(T,n,m) for k = 1:Nmpc-1]
    f = [zeros(T,n) for k = 1:Nmpc-1]

    Q = [zeros(T,n,n) for k = 1:Nmpc]
    R = [zeros(T,m,m) for k = 1:Nmpc-1]
    H = [zeros(T,m,n) for k = 1:Nmpc-1]
    q = [zeros(T,n) for k = 1:Nmpc]
    r = [zeros(T,m) for k = 1:Nmpc-1]
    
    for k = 1:Nmpc-1
        f[k] .= fd(Xref[k], Uref[k], h) - Xref[k+1]
        J = df(Xref[k], Uref[k], h)

        A[k] .= J[:,1:n]
        B[k] .= J[:,1+n:end]
        Q[k] .= Diagonal(Qd)
        R[k] .= Diagonal(Rd)
    end
    Q[Nmpc] .= Diagonal(Qd)
    return A,B,f, Q,R,H,q,r
end

function tvlqr(fd::Function, df::Function, Xref, Uref, Qd, Rd, x0)
    K,d,P,p = tvlqr(build_tvlqr(fd, df, Xref, Uref, Qd, Rd)...)

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

    return X,U,Y, K,d,A,B,f
end

function dlqr(fd::Function, df::Function, Xref, Uref, Qd, Rd, h)
    T = promote_type(eltype(Qd), eltype(Rd))
    n = length(Xref[1])
    m = length(Uref[1])
    Nmpc = length(Xref)
    A = [zeros(T,n,n) for k = 1:Nmpc-1]
    B = [zeros(T,n,m) for k = 1:Nmpc-1]
    f = [zeros(T,n) for k = 1:Nmpc-1]

    Q = [zeros(T,n,n) for k = 1:Nmpc]
    R = [zeros(T,m,m) for k = 1:Nmpc-1]
    H = [zeros(T,m,n) for k = 1:Nmpc-1]
    q = [zeros(T,n) for k = 1:Nmpc]
    r = [zeros(T,m) for k = 1:Nmpc-1]
    
    X = [zeros(T,n) for k = 1:Nmpc]
    U = [zeros(T,m) for k = 1:Nmpc-1]
    Y = [zeros(T,n) for k = 1:Nmpc]

    for k = 1:Nmpc-1
        f[k] .= fd(Xref[k], Uref[k], h) - Xref[k+1]
        J = df(Xref[k], Uref[k], h)

        A[k] .= J[:,1:n]
        B[k] .= J[:,1+n:end]
        Q[k] .= Diagonal(Qd)
        R[k] .= Diagonal(Rd)
    end
    Q[Nmpc] .= Diagonal(Qd)
    ∂ = dlqr(A,B,f, Q,R,H,q,r)

    return ∂
end

function tvlqr_get_control(fd::Function, df::Function, Xref, Uref, Qd, Rd, h, x0)
    _,dU = tvlqr(fd, df, Xref, Uref, Qd, Rd, h, x0)
    return Uref[1] + dU[1]
end

function nonlinear_rollout(f, K,d, xref, uref, x0, h)
    N = length(K) + 1
    X = [copy(x0) for k = 1:N]
    U = deepcopy(d)
    for k = 1:N-1
        dx = X[k] - xref[k]
        du = -K[k] * dx + d[k]
        U[k] = uref[k] + du
        X[k+1] = f(X[k], U[k], h)
    end
    return X,U
end

function objective(obj::TrackingObjective, theta)
    T = eltype(T)
    n,m = size(ctrl.B[1])
    Nmpc = obj.Nmpc
    Qd = theta[1:n]
    Rd = theta[n+1:end]

    
    Xref = obj.Xref
    Nref = legnth(Xref)
    xk = zeros(eltype(theta), n) 
    xk .= Xref[1]
    J = zero(eltype(theta))

    for i = 1:Nref-1

        # Calculate TVLQR 
        K,d = tvlqr(obj.f, obj.df, obj.Xref, obj.Uref, Qd, Rd)

        # Evaluate the control
        dx = xk - Xref[i]
        du = K[1] * dx + d[1]
        uk = xref[i] + du

        # Propagate the dynamics
        xk .= obj.f(xk,uk)

        # Calculate the cost
        dx = xk - xref[i+1]
        J += 0.5 * dot(dx,dx) 
    end
    return J
end

## Generate the model
model = RobotZoo.BicycleModel()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
fd(x,u,h) = RD.discrete_dynamics(dmodel, x, u, 0.0, h)
df(x,u,h) = begin
    T = promote_type(eltype(x), eltype(u))
    n,m = length(x), length(u)
    J = zeros(T, n, n + m)
    y = zeros(T, n)
    z = RD.KnotPoint{4,2}([x;u],0.0,h)
    RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dmodel, J, y, z)
    J
end 
n,m = RD.dims(model)

## Generate the reference trajectory
scotty_interp = generate_scotty_trajectory(;scale=0.10)
total_length = knots(scotty_interp).knots[end]

Nref = 501
tref = 50
h = tref / (Nref -1)
average_speed = total_length / tref

s = range(0, total_length, length=Nref)
xy_ref = scotty_interp.(s)

Xref = [zeros(n) for k = 1:Nref]
Uref = [zeros(m) for k = 1:Nref]
for k = 1:Nref
    p = xy_ref[k]
    v = 0.0
    if k == 1 
        pn = xy_ref[k+1]
        theta = atan(pn[2] - p[2], pn[1] - p[1]) 
        v = norm(pn - p) / h
    elseif k < Nref
        p_prev = xy_ref[k-1]
        θ_prev = Xref[k-1][3]
        p_next = xy_ref[k+1]

        d1 = [p - p_prev; 0]
        d2 = [p_next - p; 0]
        phi = cross(d1, d2)
        dtheta = asin(phi[3] / (norm(d1) * norm(d2)))
        theta = θ_prev + dtheta 
        v = norm(p_next - p) / h
    else
        theta = Xref[k-1][3] 
    end
    Xref[k] = [p; theta; 0.0]
    Uref[k] = [v; 0.0]
end

## Create a dynamically feasible reference
# Uref = [[1.0, 10*sin(k/pi)] for k = 1:Nref-1]
# Xref = [zeros(n) for k = 1:Nref]
# for k = 1:Nref-1
#     Xref[k+1] = fd(Xref[k], Uref[k], h)
# end

function simulate(fd, df, Xref, Uref, Qd, Rd, h, x0=copy(Xref[1]); Nmpc=21, tsim=h*(length(Xref) - Nmpc))
    T = promote_type(eltype(Qd), eltype(Rd))
    times = range(0, tsim, step=h)
    n = length(Xref[1])
    m = length(Uref[1])
    Nsim = length(times)
    Xsim = [zeros(T,n) * NaN for t in times]
    Usim = [zeros(T,m) for k = 1:Nsim-1]
    Xmpc = [[zeros(T,n) for k = 1:Nmpc] for i = 1:Nsim-1]

    mpc_inds = 1:Nmpc

    Xsim[1] .= x0
    for k = 1:Nsim - 1
        xref = view(Xref, mpc_inds)
        uref = view(Uref, mpc_inds)
        dx = Xsim[k] - xref[1]
        dX,dU = tvlqr(fd, df, xref, uref, Qd, Rd, h, dx)
        Xmpc[k] = xref .+ dX

        Usim[k] = uref[1] + dU[1]
        Xsim[k+1] = fd(Xsim[k], uref[1] + dU[1], h)
        mpc_inds = mpc_inds .+ 1
    end
    Xsim, Usim, Xmpc
end

## MPC Params
Qd = [1.0,1.0,1.001,1.001] 
Rd = [0.1, 0.1] 
Nmpc = 21
kstart = 300
dx0 = Float64[0.0,1.0,0,0]
x = Xref[kstart] + dx0
mpc_inds = (1:Nmpc) .+ (kstart-1)


## Step through Solve
xref = view(Xref, mpc_inds)
uref = view(Uref, mpc_inds)
dx = x - xref[1]
dX,dU,dY, K,d,A,B,f = tvlqr(fd, df, xref, uref, Qd, Rd, h, dx)

# Check optimality conditions
dyn_err = map(1:Nmpc-1) do k
    A[k] * dX[k] + B[k] * dU[k] + f[k] - dX[k+1]
end
norm(dyn_err)

stat_x = map(1:Nmpc-1) do k
    Diagonal(Qd) * dX[k] - dY[k] + A[k]'dY[k+1]
end
norm(stat_x)
stat_u = map(1:Nmpc-1) do k
    Diagonal(Rd) * dU[k] + B[k]'dY[k+1]
end
norm(stat_u)

# Plot the solution
Xmpc = xref .+ dX
p = traj2(Xref, size=(800,800), aspect_ratio=:equal)
# p = traj2(xref, size=(800,800), aspect_ratio=:equal)
traj2!(Xmpc, lw=2)

xref[1]
Xnl,Unl = nonlinear_rollout(fd, K,d, xref, uref, x, h)
traj2!(Xnl, lw=2)
display(p)

mpc_inds = mpc_inds .+ 1
x = fd(x, uref[1] + dU[1], h)

#############################################
# Diff through the closed-loop rollout
#############################################
function objective(theta)
    Qd = theta[1:n]
    Rd = theta[n+1:end]
    Xsim, Usim = simulate(fd, df, Xref, Uref, Qd, Rd, h; tsim=20.0)
    Nsim = length(Xsim)
    mapreduce(+,1:Nsim) do k
        dx = Xsim[k] - Xref[k]
        dot(dx,dx)
    end / 2Nsim
end

# Parameter derivatives
dQ_dtheta = [reduce(hcat,evec(n*n,i + (i-1)*n) for i = 1:n) zeros(n*n, m)]
dR_dtheta = [zeros(m*m, n) reduce(hcat, evec(m*m, i + (i-1)*m) for i = 1:m)]

# Simulate the system
Xsim, Usim = simulate(fd, df, Xref, Uref, theta[1:n], theta[n+1:end], h; tsim=20.0)
Nsim = length(Xsim)
Jsim = map(1:Nsim-1) do k
    df(Xsim[k], Usim[k], h)
end
Asim = [J[:,1:n] for J in Jsim]
Bsim = [J[:,1+n:end] for J in Jsim]


# Build TVLQR data
mpc_inds = 1:Nmpc
xref = view(Xref, mpc_inds)
uref = view(Uref, mpc_inds)
A,B,f, Q,R,H,q,r = build_tvlqr(fd, df, xref, uref, Qd, Rd)
dx = Xsim[1] - xref[1]

# Use ForwardDiff to diff TVLQR 
du1_dQd = ForwardDiff.jacobian(Qd) do Qd
    Q_ = similar.(Q,eltype(Qd))
    for i = 1:Nmpc
        Q_[i] .= Diagonal(Qd) 
    end
    # Q_[k] .= Qk
    K_,d_ = tvlqr(A,B,f, Q_,R,H, q,r)
    -K_[1] * dx + d_[1]
end
du1_dRd = ForwardDiff.jacobian(Rd) do Rd
    R_ = similar.(R,eltype(Rd))
    for i = 1:Nmpc-1
        R_[i] .= Diagonal(Rd) 
    end
    K_,d_ = tvlqr(A,B,f, Q,R_,H, q,r)
    -K_[1] * dx + d_[1]
end
dU1 = ForwardDiff.jacobian(theta->simulate(fd, df, Xref, Uref, theta[1:n], theta[n+1:end], h; tsim=20.0)[2][1], theta)

# Get the TVLQR derivatives
∂1 = dlqr(A,B,f, Q,R,H,q,r)
du1_dQ = mapreduce(+,2:Nmpc) do k
    ∂1.dP[k-1] * ∂1.dP_dQ[k] + ∂1.dp[k-1] * ∂1.dp_dQ[k]
end
du1_dR = mapreduce(+,1:Nmpc-1) do k
    if k > 1
        ∂1.dP[k-1] * ∂1.dP_dR[k] + ∂1.dp[k-1] * ∂1.dp_dR[k]
    else
        ∂1.dd_dR
    end
end
@test du1_dQ * dQ_dtheta ≈ [du1_dQd zeros(m,m)]
@test du1_dR * dR_dtheta ≈ [zeros(m,n) du1_dRd]
du1_dtheta = du1_dQ * dQ_dtheta + du1_dR * dR_dtheta

@test dU1 ≈ du1_dtheta

#############################################
# Next time step
#############################################
mpc_inds = 1 .+ (1:Nmpc) 
xref = view(Xref, mpc_inds)
uref = view(Uref, mpc_inds)

# Use ForwardDiff to get TVLQR derivatives
A,B,f, Q,R,H,q,r = build_tvlqr(fd, df, xref, uref, Qd, Rd)
dx = Xsim[2] - xref[1]
du1_dQd = ForwardDiff.jacobian(Qd) do Qd
    Q_ = similar.(Q,eltype(Qd))
    for i = 1:Nmpc
        Q_[i] .= Diagonal(Qd) 
    end
    K_,d_ = tvlqr(A,B,f, Q_,R,H, q,r)
    -K_[1] * dx + d_[1]
end
du1_dRd = ForwardDiff.jacobian(Rd) do Rd
    R_ = similar.(R,eltype(Rd))
    for i = 1:Nmpc-1
        R_[i] .= Diagonal(Rd) 
    end
    K_,d_ = tvlqr(A,B,f, Q,R_,H, q,r)
    -K_[1] * dx + d_[1]
end
dX2 = ForwardDiff.jacobian(theta->simulate(fd, df, Xref, Uref, theta[1:n], theta[n+1:end], h; tsim=20.0)[1][2], theta)
dU2 = ForwardDiff.jacobian(theta->simulate(fd, df, Xref, Uref, theta[1:n], theta[n+1:end], h; tsim=20.0)[2][2], theta)

# Calculate TVLQR derivatives
∂2 = dlqr(fd, df, xref, uref, Qd, Rd, h)
dd_dQ_2 = mapreduce(+,2:Nmpc) do k
    ∂2.dP[k-1] * ∂2.dP_dQ[k] + ∂2.dp[k-1] * ∂2.dp_dQ[k]
end
dd_dR_2 = mapreduce(+,1:Nmpc-1) do k
    if k > 1
        ∂2.dP[k-1] * ∂2.dP_dR[k] + ∂2.dp[k-1] * ∂2.dp_dR[k]
    else
        ∂2.dd_dR
    end
end
dK_dQ_2 = mapreduce(+,2:Nmpc) do k
    ∂2.dP_K[k-1] * ∂2.dP_dQ[k]
end
dK_dR_2 = mapreduce(+,1:Nmpc-1) do k
    if k > 1
        ∂2.dP_K[k-1] * ∂2.dP_dR[k]
    else
        ∂2.dK_dR
    end
end

@test dX2 ≈ Bsim[1] * du1_dtheta
du1_dQ_2 = (-kron(dx',I(m)) * dK_dQ_2 + dd_dQ_2) * dQ_dtheta
du1_dR_2 = (-kron(dx',I(m)) * dK_dR_2 + dd_dR_2) * dR_dtheta
@test du1_dQ_2 ≈ [du1_dQd zeros(m,m)]
@test du1_dR_2 ≈ [zeros(m,n) du1_dRd]
du1_dtheta_2 = du1_dQ_2 + du1_dR_2 - ∂2.K[1] * dX2

@test du1_dtheta_2 ≈ dU2

#############################################
## Calculate derivatives of CL trajectory
#############################################
dX_dθ = [zeros(n,n+m) for k = 1:Nsim]
dU_dθ = [zeros(m,n+m) for k = 1:Nsim]

dX_dθ[1] .= 0
for i = 1:Nsim-1
    mpc_inds = (i-1) .+ (1:Nmpc) 
    xref = view(Xref, mpc_inds)
    uref = view(Uref, mpc_inds)
    dx = Xsim[i] - xref[1]                   # state error at first time step
    ∂ = dlqr(fd, df, xref, uref, Qd, Rd, h)  # calculate derivatives of TVLQR

    # Calculate derivatives wrt parameters
    # Since the parameters apply at each time step, sum the derivatives of each time step
    dd_dQ = mapreduce(+,2:Nmpc) do k
        ∂.dP[k-1] * ∂.dP_dQ[k] + ∂.dp[k-1] * ∂.dp_dQ[k]
    end
    dd_dR = mapreduce(+,1:Nmpc-1) do k
        if k > 1
            ∂.dP[k-1] * ∂.dP_dR[k] + ∂.dp[k-1] * ∂.dp_dR[k]
        else
            ∂.dd_dR
        end
    end
    dK_dQ = mapreduce(+,2:Nmpc) do k
        ∂.dP_K[k-1] * ∂.dP_dQ[k]
    end
    dK_dR = mapreduce(+,1:Nmpc-1) do k
        if k > 1
            ∂.dP_K[k-1] * ∂.dP_dR[k]
        else
            ∂.dK_dR
        end
    end

    # Calculate total derivative of u1 wrt the parameters θ
    du_dQ = (-kron(dx',I(m)) * dK_dQ + dd_dQ) * dQ_dtheta
    du_dR = (-kron(dx',I(m)) * dK_dR + dd_dR) * dR_dtheta
    dU_dθ[i] = du_dQ + du_dR - ∂.K[1] * dX_dθ[i] 

    # Derivative of the state wrt the parameters θ
    dX_dθ[i+1] = Asim[i] * dX_dθ[i] + Bsim[i] * dU_dθ[i]
end
@test dU_dθ[2] ≈ du1_dtheta_2

dXN = ForwardDiff.jacobian(theta->simulate(fd, df, Xref, Uref, theta[1:n], theta[n+1:end], h; tsim=20.0)[1][Nsim], theta)
@test dX_dθ[Nsim] ≈ dXN

dtheta = mapreduce(+,1:Nsim) do i
    dx = Xsim[i] - Xref[i]
    dX_dθ[i]'dx
end / Nsim

dtheta_ad = ForwardDiff.gradient(objective, theta)
# Derivative of entire objective

Qd = [1.0,1.0,10.100,10.100] 
Rd = [0.01, 0.01] 
theta = [Qd; Rd]

alpha = 0.1
pos(x) = max(zero(x), x)
for i = 1:100
    J = objective(theta)
    # dtheta = FiniteDiff.finite_difference_gradient(objective, theta)
    dtheta = ForwardDiff.gradient(objective, theta)
    cost_decrease = false
    alpha = 0.5
    dJ = Inf 
    for j = 1:20
        theta_bar = pos.(theta - alpha * dtheta)
        Jbar = Inf
        try
            Jbar = objective(theta_bar)
        catch
            Jbar = Inf
        end
        if Jbar < J
            dJ = J - Jbar
            theta .= theta_bar
            cost_decrease = true
            break
        else
            alpha *= 0.5
        end
    end
    @printf("Iter %3d: J = %10.5f, ||dJ|| = %10.5g, dJ = %10.4g, alpha = %10.4g\n", i, J, norm(dtheta), dJ, alpha)
    if dJ < 1e-6
        @info "Cost not decreasing much"
        break
    end
    if !cost_decrease
        @warn "Cost not decreased"
        break
    end
end

objective(theta)
theta
Qd
theta
Xsim, Usim = simulate(fd, df, Xref, Uref, Qd, Rd, h; tsim=20.0)
Xsim2, Usim2 = simulate(fd, df, Xref, Uref, theta[1:n], theta[n+1:end], h; tsim=20.0)
Nsim = length(Xsim)
traj2(Xref, size=(800,800))
traj2!(Xsim)
traj2!(Xsim2)

norm(Xsim - Xref[1:Nsim])^2 / Nsim
norm(Xsim2 - Xref[1:Nsim])^2 / Nsim