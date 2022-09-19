using RecipesBase 

"""
    traj2(x, y)
    traj2(X; xind=1, yind=2)
    traj2(Z::AbstractTrajectory; xind=1, yind=2)
Plot a 2D state trajectory, given the x and y positions of the robot. If given
a state trajectory, the use optional `xind` and `yind` arguments to provide the
location of the x and y positions in the state vector (typically 1 and 2, respectively).
"""
@userplot Traj2

@recipe function f(traj::Traj2; xind=1, yind=2)
    # Process input
    #   Input a vector of static vectors
    if length(traj.args) == 1 && (traj.args[1] isa AbstractVector{<:AbstractVector{<:Real}})
        X = traj.args[1]
        xs = [x[xind] for x in X]
        ys = [x[yind] for x in X]
    #   Input the x and y vectors independently
    elseif length(traj.args) == 2 && 
            traj.args[1] isa AbstractVector{<:Real} &&
            traj.args[2] isa AbstractVector{<:Real}
        xs = traj.args[1]
        ys = traj.args[2]
    else
        throw(ArgumentError("Input must either be a Vector of StaticVector's, or two Vectors of positions"))
    end
    # Defaults
    xguide --> "x"
    yguide --> "y"
    label --> :none
    # Plot x,y
    (xs,ys)
end