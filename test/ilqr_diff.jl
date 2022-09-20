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

K,d, = tvlqr(A,B,f, lxx,luu,lux,lx,lu)

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
xilqr, uilqr = SimpleAltro.ilqr(dyn,jac, x0_zero,u0_zero,h, Q,R,H,q,r)


##
traj2(Xref, size=(800,800), aspect_ratio=:equal)
traj2!(x0_zero, lw=2)
traj2!(xilqr, lw=2)