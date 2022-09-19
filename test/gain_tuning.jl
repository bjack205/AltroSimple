using LinearAlgebra
import SimpleAltro: tvlqr, dlqr 
using RobotZoo
import RobotDynamics as RD
using Plots

include("recipes.jl")
include("scotty.jl")

struct TrackingObjective{T}
    f::Function     # dynamics function
    df::Function    # dynamics Jacobian

    # Reference trajectory
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}

    # MPC params
    Nmpc::Int
end

function tvlqr(fd::Function, df::Function, Xref, Uref, Qd, Rd, h, x0)
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
    K,d,P,p = tvlqr(A,B,f, Q,R,H,q,r)

    X[1] .= x0
    for k = 1:Nmpc-1
        Y[k] = P[k] * X[k] + p[k]
        U[k] = -K[k] * X[k] + d[k]
        X[k+1] = A[k] * X[k] + B[k] * U[k] + f[k]
    end
    Y[Nmpc] = P[Nmpc] * X[Nmpc] + p[Nmpc]

    return X,U,Y, K,d,A,B,f
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

function simulate(fd, df, Xref, Uref, h, Qd, Rd, x0=copy(Xref[1]); Nmpc=21, tsim=h * (length(Xref) - 21))
    m = length(Uref[1])
    times = range(0, tsim, step=h)
    Nsim = length(times)
    Xsim = [copy(x0) for t in times]
    Usim = [zeros(m) for t in times]
    mpc_inds = 1:Nmpc
    for k = 1:Nsim-1
        # Evaluate controller 
        xref = view(Xref, mpc_inds)
        uref = view(Uref, mpc_inds)
        dx = Xsim[k] - Xref[k]
        dX,dU = tvlqr(fd, df, xref, uref, Qd, Rd, h, dx)

        Usim[k] = dU[1] + uref[1]
        Xsim[k+1] = fd(Xsim[k], Usim[k], h)

        mpc_inds = mpc_inds .+ 1
    end
    Xsim, Usim
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
Xref

Nref = 2001
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
Uref = [[1.0, 10*sin(k/pi)] for k = 1:Nref-1]
Xref = [zeros(n) for k = 1:Nref]
for k = 1:Nref-1
    Xref[k+1] = fd(Xref[k], Uref[k], h)
end

##
dx0 = Float64[0.0,0.0,0,0]
x = Xref[1] + dx0
mpc_inds = 1:Nmpc


## Solve
Nmpc = 21
Qd = [1.0,1.0,0.001,0.001] 
Rd = fill(0.001,m)
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

# fd(x, uref[1] + dU[1], h)
# A[1] * dX[1] + B[1] * dU[1] + f[1] + xref[2] 

dU[1]


##

# Xsim, Usim = simulate(fd, df, Xref, Uref, h, Qd, Rd)
# traj2(Xref, size=(800,800))
# traj2!(Xsim)