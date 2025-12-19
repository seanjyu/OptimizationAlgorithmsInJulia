module UnivariatePlottingModule
    using ..GradientEstimatorInterface: GradientEstimator, gradient
    using Plots

    export plotFunction, plotGradients!, plotLineSearchPoints!
 
    function plotFunction(f, xLow::Number, xHigh::Number, interval::Number; 
                        linewidth=2, color=:blue, label="f(x)", title="")
        x_range = xLow:interval:xHigh
        return plot(x_range, f(x_range), 
                linewidth=linewidth, 
                color=color, 
                label=label,
                title=title)
    end

    function plotGradients!(f, gradEstimator::GradientEstimator, xLow::Number, xHigh::Number, interval::Number;
                                linewidth=2, color=:red, label="grad f", base_arrow_dx = 1)
        x_points = xLow:interval:xHigh
        grad_values = [gradient(gradEstimator, f, x)[1] for x in x_points]
        y_points = f.(x_points)
        for (i, x) in enumerate(x_points)
            y = y_points[i]
            curSlope = grad_values[i]

            arrow_length_scale = abs(curSlope)

            # Create direction vector (normalized)
            direction_magnitude = sqrt(1 + curSlope^2)
            dx = (base_arrow_dx * arrow_length_scale) / direction_magnitude
            dy = (base_arrow_dx * arrow_length_scale * curSlope) / direction_magnitude
            
            # Only label the first arrow
            current_label = (i == 1) ? label : ""
            
            plot!([x - dx/2, x + dx/2], [y[1] - dy/2, y[1] + dy/2], 
                arrow=true, color=color, linewidth=linewidth, label=current_label)
        end
    end

    function plotLineSearchPoints!(f, path; linewidth=2, color=:red, label="Descent Path")
        yValues = [f(x)[1] for x in path]
        # original version with just markers
        # plot!(path, yValues, 
        #         linewidth=linewidth, 
        #         color=color, 
        #         marker=:circle,
        #         markersize=4,
        #         label=label)
        # Remove the marker plot, add arrows instead
        for i in 1:(length(path)-1)
            x_start, x_end = path[i], path[i+1]
            y_start, y_end = yValues[i], yValues[i+1]
            
            plot!([x_start, x_end], [y_start, y_end], 
                arrow=true, 
                linewidth=linewidth, 
                color=color, 
                alpha=0.8,
                label=(i==1 ? label : ""))  # Only label the first one
        
        end
    end
end

module MultivariatePlottingModule
    using ..GradientEstimatorInterface: GradientEstimator, gradient
    using Plots

    export plot2DContourFunction, plotGradients2DContour!, plotLineSearchPoints2DContour!, plot3D, plotLineSearchPoints3D!

    function plot2DContourFunction(f, x1Low::Number, x1High::Number, x2Low::Number, x2High::Number, interval::Number;
                                x1Label="x₁", x2Label="x₂", title="", fill=false, colorBar = true)
      
        # Calculate spans - number of points need to be same for both axes to plot contour
        x1_span = x1High - x1Low
        x2_span = x2High - x2Low
    
        # If span not the same use the maximum number of points for both axes
        if x1_span != x2_span
            n_points = max(round(Int, x1_span / interval) + 1, round(Int, x2_span / interval) + 1)
            
            x1Range = range(x1Low, x1High, length=n_points)
            x2Range = range(x2Low, x2High, length=n_points)
        else
            x1Range = x1Low:interval:x1High
            x2Range = x2Low:interval:x2High
        end

        # Evaluate function at each grid point
        # Z = [f([x1, x2]) for x1 in x1Range, x2 in x2Range]
        Z = [f([x1, x2]) for x2 in x2Range, x1 in x1Range]

        # create 2D contour plot
	    return contour(x1Range, x2Range, Z,
            xlabel=x1Label, ylabel=x2Label, 
            title=title, fill=fill, colorbar = colorBar, aspect_ratio=:auto)
 
    end

    function plotGradients2DContour!(f, gradEstimator::GradientEstimator, 
                                    x1Low::Number, x1High::Number, 
                                    x2Low::Number, x2High::Number, 
                                    interval::Number, contour_level::Number; 
                                    tolerance=0.1, scale_factor=0.1, 
                                    color=:red, alpha=0.7)

        x1Range = x1Low:interval:x1High
        x2Range = x2Low:interval:x2High
        x1Interior = x1Range[2:end-1]
        x2Interior = x2Range[2:end-1]

        # Generate mesh grid points
        x1_points = repeat(x1Interior, inner=length(x2Interior))
        x2_points = repeat(x2Interior, outer=length(x1Interior))

        # Compute gradients
        grads = [gradient(gradEstimator, f, [x, y]) for (x, y) in zip(x1_points, x2_points)]
        U = [g[1] for g in grads]
        V = [g[2] for g in grads]

        # Scale gradients for visibility
        U_scaled = -U .* scale_factor
        V_scaled = -V .* scale_factor

        # Plot quiver
        quiver!(x1_points, x2_points, 
                quiver=(U_scaled, V_scaled),
                color=color, alpha=alpha)
    end

    function plot3D()
        #TODO
    end

    function plotLineSearchPoints2DContour!(path; linewidth=2, color=:red, label="Descent Path")
        x_coords = [point[1] for point in path]
        y_coords = [point[2] for point in path]
        for i in 1:1:(length(path)-1)
            x_start, x_end = x_coords[i], x_coords[i+1]
            y_start, y_end = y_coords[i], y_coords[i+1]
            
            plot!([x_start, x_end], [y_start, y_end], 
                arrow=true, 
                linewidth=linewidth, 
                color=color, 
                alpha=0.8,
                label=(i==1 ? label : ""))
        end
    end

    function plotLineSearchPoints3D!()
        #TODO
    end

end