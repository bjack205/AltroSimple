using Interpolations

function generate_scotty_trajectory(;scale=1.0)
    scottyraw = [
        (140, 514),
        (230, 540),
        (360, 566),
        (420,570),
        (470,560),
        (480,590),
        (500,600),
        (591,607),
        (602,592),
        (570,560),
        (560,510),
        (600,470),
        (630,404),
        (625,309),
        (640,223),
        (733,270),
        (755,217),
        (765,125),
        (735,62),
        (670,64),
        (610,80),
        (637,53),
        (536,71),
        (524,27),
        (497,0),
        (485,51),
        (465,26),
        (438,15),
        (442,80),
        (455,155),
        (459,210),
        (448,242),
        (430,280),
        (368,287),
        (261,277),
        (183,267),
        (179,221),
        (188,160),
        (200,116),
        (170,141),
        (149,189),
        (135,230),
        (131,265),
        (66,312),
        (23,374),
        (5,500),
        (7,540),
        (6,535),
        (17,577),
        (34,603),
        (72,616),
        (116,613),
        (138,600),
        (133,586),
        (117,577),
        (121,550),
        (140, 514),
        # (141,518),
    ]
    # Flip over x-axis
    scotty = map(scottyraw) do (x,y)
        (x*scale,-y*scale)
    end

    dist(p1,p2) = sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
    segment_lengths = [dist(scotty[i], scotty[i+1]) for i = 1:length(scotty)-1]
    s = pushfirst!(cumsum(segment_lengths), 0)
    total_length = sum(segment_lengths)
    linear_interpolation(s, collect.(scotty))
end

# using Plots
# plot(scotty, aspect_ratio=:equal)
# plot!(Tuple.(newscotty), aspect_ratio=:equal)

function scotty_traj_bicycle(; tref=50.0, Nref=501, scale=0.1)
    scotty_interp = generate_scotty_trajectory(;scale)
    total_length = knots(scotty_interp).knots[end]

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
    return Xref, Uref
end