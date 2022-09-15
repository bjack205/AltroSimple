using LinearAlgebra
using ForwardDiff
using BenchmarkTools
include("../src/cones.jl")


function dhess_dcon(cone, rho, jac_con, jac_proj, lambda_bar)
    p,n = size(jac_con)
    hess = zeros(n*n, p)
    jac1 = kron(jac_con', jac_con'jac_proj)
    jac2 = kron(jac_con'jac_proj, jac_con')
    for i = 1:p
        hess_proj = vec(∇²projection(cone, lambda_bar, even(i,n)))
        hess[:,i] .= (jac1 + jac2) * hess_proj
    end
end

evec(i,n) = insert!(zeros(n-1), i, 1) 

n = 6
p = 4
v = randn(p-1)
x = randn(n)

A = [randn(p-1,n); zeros(1,n)]
b = push!(zeros(p-1), norm(A*x) - 1e-2)
c = A*x + b
λ = randn(p)
ρ = 1.2
con(x) = A*x + b 
jac(x) = A 
λbar(x) = λ - ρ * con(x)

# Cone projection and derivatives
Pi(x) = projection(SecondOrderCone(), x)
jac_Pi(x) = ∇projection(SecondOrderCone(), x)
hess_Pi(x,b) = ∇²projection(SecondOrderCone(), x, b)

obj(x) = 1/2ρ * dot(Pi(λbar(x)), Pi(λbar(x)))
ForwardDiff.gradient(obj, x) ≈ -jac(x)'jac_Pi(λbar(x))'Pi(λbar(x))
ForwardDiff.hessian(obj, x) ≈ ρ*jac(x)'jac_Pi(λbar(x))'jac_Pi(λbar(x))*jac(x) + ρ*jac(x)'hess_Pi(λbar(x),Pi(λbar(x)))*jac(x)
dhess_dc_fd = ForwardDiff.jacobian(c->ρ*jac(x)'jac_Pi(λ-ρ*c)'jac_Pi(λ-ρ*c)*jac(x), c)  # Guass-Newton approximation

# Derivative of objective Hessian with respect to the constraint 
dhess_dc = zeros(n*n,p)
jac_kron = ρ*kron(jac(x)', jac(x)'jac_Pi(λbar(x))') + ρ*kron(jac(x)'jac_Pi(λbar(x))', jac(x)')
for i = 1:p
    dhess_dc[:,i] =  -ρ*jac_kron * vec(hess_Pi(λbar(x),evec(i,p)))
end
dhess_dc ≈ dhess_dc_fd

# Derivative of the objective gradient with respect to the constraint
dgrad_dc_fd = ForwardDiff.jacobian(c->-jac(x)'jac_Pi(λ-ρ*c)'Pi(λ-ρ*c), c)
dgrad_dc = ρ*jac(x)'hess_Pi(λbar(x), Pi(λbar(x))) + ρ*jac(x)'jac_Pi(λbar(x))'jac_Pi(λbar(x))
dgrad_dc ≈ dgrad_dc_fd
