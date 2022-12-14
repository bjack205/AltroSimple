function cost(x,u, Q,R,H,q,r)
    T = promote_type(eltype(x[1]), eltype(x[1]), eltype(Q[1]), eltype(R[1]), eltype(H[1]), 
        eltype(q[1]), eltype(r[1]))
    J = zero(T)
    N = length(x)
    for k = 1:N
        J += 0.5 * dot(x[k], Q[k], x[k]) + dot(q[k], x[k])
        if k < N
            J += 0.5 * dot(u[k], R[k], u[k]) + dot(r[k], u[k])
            J += dot(u[k], H[k], x[k])
        end
    end
    J
end

function rollout_merit(alpha::T, dyn, xref,uref,h, K,d, Q,R,H,q,r) where T
    N = length(xref)
    n = length(xref[1])
    m = length(uref[1])
    x = [zeros(T,n) for k = 1:N] 
    u = [zeros(T,m) for k = 1:N-1]

    x[1] .= xref[1]
    J = 0.5 * dot(x[1], Q[1], x[1]) + dot(q[1], x[1])
    for k = 1:N-1
        dx = x[k] - xref[k] 
        du = -K[k] * dx + alpha * d[k]
        u[k] = uref[k] + du
        x[k+1] = dyn(x[k], u[k], h)

        J += dot(u[k], H[k], x[k])
        J += 0.5 * dot(u[k], R[k], u[k]) + dot(r[k], u[k])
        J += 0.5 * dot(x[k+1], Q[k+1], x[k+1]) + dot(q[k+1], x[k+1])
    end
    J
end

function rollout_merit_derivative(alpha, dyn,jac, xref,uref,h, K,d, Q,R,H,q,r)
    T = promote_type(typeof(alpha), eltype(Q[1]), eltype(R[1]), eltype(H[1]), eltype(q[1]), eltype(r[1]))
    N = length(xref)
    n = length(xref[1])
    m = length(uref[1])
    x = [zeros(T,n) for k = 1:N] 
    u = [zeros(T,m) for k = 1:N-1]

    x[1] .= xref[1]
    dJ = zero(alpha)
    dx_da = zeros(T,n)
    J = 0.5 * dot(x[1], Q[1], x[1]) + dot(q[1], x[1])
    for k = 1:N-1
        dx = x[k] - xref[k] 
        du = -K[k] * dx + alpha * d[k]
        u[k] = uref[k] + du
        x[k+1] = dyn(x[k], u[k], h)

        J += dot(u[k], H[k], x[k])
        J += 0.5 * dot(u[k], R[k], u[k]) + dot(r[k], u[k])
        J += 0.5 * dot(x[k+1], Q[k+1], x[k+1]) + dot(q[k+1], x[k+1])

        jac_dyn = jac(x[k], u[k], h)
        A = jac_dyn[:,1:n]      # TODO: store these to use in the next backward pass
        B = jac_dyn[:,n+1:end]
        du_da = -K[k] * dx_da + d[k]
        dJ += dot(du_da, H[k], x[k])
        dJ += dot(u[k], H[k], dx_da)

        dx_da .= A*dx_da + B*du_da
        dJ += dot(u[k], R[k], du_da) + dot(r[k], du_da)
        dJ += dot(x[k+1], Q[k+1], dx_da) + dot(q[k+1], dx_da)
    end
    return J, dJ, x, u
end

function ilqr(dyn, jac, x0,u0,h, Q,R,H,q,r;
        max_iters=100,
        ls_iters=25,
        tol_cost = 1e-6,
        verbose = 1,
    )
    x = deepcopy(x0)
    u = deepcopy(u0)
    n = length(x0[1])
    m = length(u0[1])
    N = length(x0)
    T = promote_type(eltype(x[1]), eltype(x[1]), eltype(Q[1]), eltype(R[1]), eltype(H[1]), 
        eltype(q[1]), eltype(r[1]))

    # Dynamics expansion
    A = [zeros(T, n, n) for k = 1:N-1]
    B = [zeros(T, n, m) for k = 1:N-1]
    f = [zeros(T,n) for k = 1:N-1]

    # Cost expansion
    lxx = deepcopy(Q) 
    lux = deepcopy(H) 
    luu = deepcopy(R) 
    lx = [zeros(T, n) for k = 1:N]
    lu = [zeros(T, m) for k = 1:N]

    xu_hist = [(x,u)]
    local P,p
    for iter = 1:max_iters
        cost_prev = cost(x, u, Q, R, H, q, r)

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

        # Calculate TVLQR policy
        dx,du,dy, K,d,P,p = tvlqr(A,B,f, lxx,luu,lux,lx,lu, zeros(n))

        stat_x = map(1:N-1) do k
            lxx[k] * dx[k] - dy[k] + A[k]'dy[k+1] + lx[k]
        end
        sdx = norm(stat_x)

        stat_u = map(1:N-1) do k
            luu[k] * du[k] + B[k]'dy[k+1] + lu[k]
        end
        sdu = norm(stat_u)

        sx = Inf
        su = Inf

        # Forward pass
        ?? = 1.0
        c??? = 1e-4
        c??? = 0.9
        line_search_successful = false
        J0,dJ0 = rollout_merit_derivative(0.0, dyn,jac, x,u,h, K,d, Q,R,H,q,r)
        J = Inf
        dJ = Inf
        for ls_iter = 1:ls_iters
            J,dJ, xbar, ubar = rollout_merit_derivative(??, dyn,jac, x,u,h, K,d, Q,R,H,q,r)
            armijo = J ??? J0 + c??? * ?? * dJ0
            wolfe = abs(dJ) ??? c??? * abs(dJ0)
            verbose >= 2 && @printf(
                "  ls iter %2d: ?? = %10.4g J = %10.4g dJ = %10.4g armijo? %d wolfe? %d\n", 
                ls_iter, ??, J, dJ, armijo, wolfe
            )
            if armijo && wolfe
                x = deepcopy(xbar)
                u = deepcopy(ubar)
                # push!(xu_hist, (x,u))
                line_search_successful = true
                break
            else 
                ?? *= 0.5  # TODO: quadratic line search
            end
        end
        if !line_search_successful
            error("Line search failed.")
        end

        # Stationarity
        y = p  # the Lagrange multipliers converge to the gradient of the cost-to-go
        sx = mapreduce(+,1:N-1) do k
            norm(lx[k] - y[k] + A[k]'y[k+1])
        end / N

        su = mapreduce(+,1:N-1) do k
           norm(lu[k] + B[k]'y[k+1])
        end / N

        # Print log
        ??J = J0 - J
        verbose >= 1 && @printf(
            "Iter %3d: J = %10.4g ??J = %10.4e ???J = % 10.4e ?? = %6.4g sx = %6.2e, su = %6.2e\n", 
            iter, J, ??J, dJ, ??, sx, su
        )

        # Termination criteria
        if ??J < tol_cost
            @info "Converged"
            break
        end
    end
    return x,u, xu_hist
end
