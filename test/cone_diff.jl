using LinearAlgebra
using ForwardDiff
using BenchmarkTools
include("../src/cones.jl")

evec(i,n) = insert!(zeros(n-1), i, 1) 
Pi(x) = projection(SecondOrderCone(), x)
jac_Pi(x) = ∇projection(SecondOrderCone(), x)
hess_Pi(x,b) = ∇²projection(SecondOrderCone(), x, b)

n = 4
v = randn(n-1)
x = [v; norm(v) - 1e-2]

obj(x) = 1/2 * dot(Pi(x), Pi(x))
ForwardDiff.gradient(obj, x) ≈ jac_Pi(x)'Pi(x)
ForwardDiff.hessian(obj, x) ≈ jac_Pi(x)'jac_Pi(x) + hess_Pi(x,Pi(x))
hess2_fd = ForwardDiff.jacobian(x->jac_Pi(x)'jac_Pi(x), x)

# Derivative of objective Hessian with respect to x
hess2 = zeros(n*n,n)
for i = 1:n
    hess2[:,i] = kron(I(n), jac_Pi(x)') * vec(hess_Pi(x,evec(i,n))) + kron(jac_Pi(x)', I(n)) * vec(hess_Pi(x, evec(i,n))')
end
hess2 ≈ hess2_fd


jac_Pi(x)'jac_Pi(x)
hess_Pi(x,Pi(x))
Pi(x)
∇²projection(SecondOrderCone(), x, evec(2,n))

hess = ForwardDiff.jacobian(jac_Pi, x)
ForwardDiff.jacobian(x->hess_Pi(x,Pi(x)))

reshape(hess[:,2], 4,4)
