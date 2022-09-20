using LinearAlgebra
using SimpleAltro
import SimpleAltro: tvlqr, dlqr 
using RobotZoo
using FiniteDiff
using ForwardDiff
import RobotDynamics as RD
using Plots
using Printf
using Test

include("scotty.jl")
include("recipes.jl")

## Generate the model
model = RobotZoo.BicycleModel()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
dyn(x,u,h) = RD.discrete_dynamics(dmodel, x, u, 0.0, h)
jac(x,u,h) = begin
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
Nref = 501
tref = 50.0
h = tref / (Nref - 1)
Xref,Uref = scotty_traj_bicycle(;Nref, tref)

## MPC Params 
Qd = [1.0, 1.0, 1e-6, 0.1]
Rd = [0.1, 0.01]
kstart = 1
Nmpc = 31
mpc_inds = (kstart - 1) .+ 1:Nmpc

## Cost matrices
xref = view(Xref, mpc_inds)
uref = view(Uref, mpc_inds)
Q = [diagm(Qd) for k = 1:Nmpc]
R = [diagm(Rd) for k = 1:Nmpc-1]
H = [zeros(m,n) for k = 1:Nmpc-1]
q = map(k->-Qd .* xref[k], 1:Nmpc)
r = map(k->-Rd .* uref[k], 1:Nmpc-1)

## Initial trajectory
jacs = [jac(x,u,h) for (x,u) in zip(xref, uref)]
A = [J[:,1:n] for J in jacs]
B = [J[:,1+n:end] for J in jacs]
f = [dyn(xref[k],uref[k],h) - xref[k+1] for k in 1:Nmpc-1] 

K,d = tvlqr(A,B,f, Q,R,H,q*0,r*0)
x0 = [zeros(n) for k = 1:Nmpc]
u0 = [zeros(m) for k = 1:Nmpc-1]
x0[1] .= xref[1]
for k = 1:Nmpc-1
    dx = x0[k] - xref[k]
    du = -K[k] * dx + d[k]
    u0[k] = uref[k] + du
    x0[k+1] = dyn(x0[k], u0[k], h)
end

x0_zero = deepcopy(x0)
u0_zero = (Uref[mpc_inds])[1:Nmpc-1]
x0_zero[1] .= x0[1]
for k = 1:Nmpc-1
    x0_zero[k+1] = dyn(x0_zero[k], u0_zero[k], h)
end

## Run first iteration

# Dynamics expansion
T = Float64
N = Nmpc
A = [zeros(T, n, n) for k = 1:N-1]
B = [zeros(T, n, m) for k = 1:N-1]
f = [zeros(T,n) for k = 1:N-1]

# Cost expansion
lxx = deepcopy(Q) 
lux = deepcopy(H) 
luu = deepcopy(R) 
lx = [zeros(T, n) for k = 1:N]
lu = [zeros(T, m) for k = 1:N]

x = deepcopy(x0)
u = deepcopy(u0)

# Calculate Dynamics Jacobians
for k = 1:N-1
    J = jac(x[k], u[k], h)
    A[k] .= J[:,1:n]
    B[k] .= J[:,1+n:end]
end

# Calculate Cost expansion
for k = 1:N
    lx[k] = Q[k] * x[k] + q[k]
    if k < N
        lu[k] = R[k] * u[k] + r[k]
    end
end

K,d,P,p = tvlqr(A,B,f, lxx,luu,lux,lx,lu)
dx,du,dy = tvlqr(A,B,f, lxx,luu,lux,lx,lu, zeros(n))
x2 = x + dx
u2 = u + du

stat_x = map(1:Nmpc-1) do k
    -dy[k] + A[k]'dy[k+1] + lx[k]
end
norm(stat_x) / Nmpc

stat_u = map(1:Nmpc-1) do k
    B[k]'dy[k+1] + lu[k]
end
norm(stat_u) / Nmpc

stat_x = map(1:Nmpc-1) do k
    lxx[k] * dx[k] - dy[k] + A[k]'dy[k+1] + lx[k]
end
@test norm(stat_x) < sqrt(eps()) 

stat_u = map(1:Nmpc-1) do k
    luu[k] * du[k] + B[k]'dy[k+1] + lu[k]
end
@test norm(stat_u) < sqrt(eps()) 

res = let k = 1
    Qxx = lxx[k] + A[k]'P[k+1]*A[k]
    Qux = lux[k] + B[k]'P[k+1]*A[k]
    Quu = luu[k] + B[k]'P[k+1]*B[k]
    Qx = lx[k] + A[k]'*(P[k+1] * f[k] + p[k+1])
    Qu = lu[k] + B[k]'*(P[k+1] * f[k] + p[k+1])

    # lu[k] + B[k]'P[k+1]*(dyn(x2[k],u2[k],h) - x[k]) + B[k]'p[k+1]
    Quu * du[k] + Qux * dx[k] + lu[k] + B[k]'*(P[k+1] * (dyn(x2[k],u2[k],h) - x2[k+1]) + p[k+1])
    # R[k] * u2[k] + r[k] + B[k]'P[k+1] * (dyn(x2[k],u2[k],h) - x[k]) + B[k]'p[k+1]
end
@show res

f1 = dyn(x2[1],u2[1],h)
B1 = jac(x2[1],u2[1],h)[:,n+1:end]
R[1] * u2[1] + r[1] + B1'P[2]*f1 + B1'p[2] 

# Check merit function derivative
xref = copy(x)
uref = copy(u)
cost_prev = SimpleAltro.cost(x,u, Q,R,H,q,r)
a = 0.001
cost_new = SimpleAltro.rollout_merit(a, dyn, xref,uref,h, K,d, Q,R,H,q,r)
J,dJ,xbar,ubar = SimpleAltro.rollout_merit_derivative(a, dyn,jac, xref,uref,h, K,d, Q,R,H,q,r)
dJ_ad = ForwardDiff.derivative(a->SimpleAltro.rollout_merit(a, dyn,xref,uref,h, K,d, Q,R,H,q,r), a)

@test cost_new < cost_prev
@test J ≈ cost_new
@test dJ ≈ dJ_ad

## Try running iLQR
SimpleAltro.ilqr(dyn,jac, x0,u0,h, Q,R,H,q,r)
xilqr, uilqr, xu_hist = SimpleAltro.ilqr(dyn,jac, x0_zero,u0_zero,h, Q,R,H,q,r)
sx,su = SimpleAltro.ilqr(dyn,jac, x0_zero,u0_zero,h, Q,R,H,q,r)

SimpleAltro.ilqr(dyn,jac, xu_hist[end-1][1],xu_hist[end-1][2],h, Q,R,H,q,r)
xu_hist[1] == (x0_zero, u0_zero)

SimpleAltro.stationarity(dyn,jac, xilqr, uilqr,h, )

###
function control_policy(theta, Nmpc, i; x0=x0, u0=u0)
    global Xref
    global Uref
    global dyn
    global jac

    n = length(Xref[1])
    m = length(Uref[1])

    Qd = theta[1:n]
    Rd = theta[n+1:end]

    mpc_inds = (i-1) .+ (1:Nmpc)
    xref = view(Xref, mpc_inds)
    uref = view(Uref, mpc_inds)

    T = eltype(theta)
    Q = [zeros(T,n,n) for k = 1:Nmpc]
    R = [zeros(T,m,m) for k = 1:Nmpc-1]
    H = [zeros(T,m,n) for k = 1:Nmpc-1]
    q = [zeros(T,n) for k = 1:Nmpc]
    r = [zeros(T,m) for k = 1:Nmpc-1]
    
    for k = 1:Nmpc-1
        Q[k] .= Diagonal(Qd)
        R[k] .= Diagonal(Rd)
        q[k] .= -Qd .* xref[k]
        r[k] .= -Rd .* uref[k]
    end
    Q[Nmpc] .= Diagonal(Qd)
    q[Nmpc] .= -Qd .* xref[Nmpc]

    SimpleAltro.ilqr(dyn,jac, x0,u0,h, Q,R,H,q,r)
end

theta = [Qd; Rd]
control_policy(theta, Nmpc, 1, x0=x0_zero, u0=u0_zero)
control_policy(theta, Nmpc, 1, x0=xu_hist[end-1][1], u0=xu_hist[end-1][2])

ForwardDiff.jacobian(theta->control_policy(theta, Nmpc, 1; x0=xu_hist[end-1][1], u0=xu_hist[end-1][2]), theta)
ForwardDiff.jacobian(theta->control_policy(theta, Nmpc, 1; x0=x0_zero, u0=u0_zero), theta)

##
traj2(Xref, size=(800,800), aspect_ratio=:equal)
traj2!(x0_zero, lw=2)
traj2!(xilqr, lw=2)
