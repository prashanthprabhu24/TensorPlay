import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def visualize_gradient_descent(func_expr, start_point, learning_rate=0.1, iterations=50):
    """
    Parses a string function f(x, y), performs gradient descent, and visualizes the path.
    """
    print(f"--- Running Gradient Descent for f(x, y) = {func_expr} ---")

    # 1. Define symbols and parse the function expression
    x, y = sp.symbols('x y')
    try:
        f = sp.sympify(func_expr)
    except Exception as e:
        print(f"Error parsing function: {e}")
        return

    # 2. Calculate exact partial derivatives using calculus (the gradient)
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    # 3. Convert SymPy expressions to NumPy functions for fast numerical evaluation
    f_num = sp.lambdify((x, y), f, 'numpy')
    grad_x_num = sp.lambdify((x, y), df_dx, 'numpy')
    grad_y_num = sp.lambdify((x, y), df_dy, 'numpy')

    # 4. Initialize history tracking for the visualization
    x_val, y_val = float(start_point[0]), float(start_point[1])
    history = [(x_val, y_val, f_num(x_val, y_val))]

    # 5. Perform the Gradient Descent Loop
    for i in range(iterations):
        # Calculate gradients at the current position
        try:
            grad_x = grad_x_num(x_val, y_val)
            grad_y = grad_y_num(x_val, y_val)
        except Exception as e:
            print(f"Math error at iteration {i}: {e}. Try a smaller learning rate.")
            break

        # Update weights (take a step down the gradient)
        x_val = x_val - learning_rate * grad_x
        y_val = y_val - learning_rate * grad_y

        # Save the new point to our history
        z_val = f_num(x_val, y_val)
        history.append((x_val, y_val, z_val))

        # Stop early if the numbers get too massive (divergence)
        if abs(x_val) > 1e6 or abs(y_val) > 1e6:
            print("Algorithm diverged! The learning rate is likely too high.")
            break

    history = np.array(history)
    print(f"Started at: x={start_point[0]:.3f}, y={start_point[1]:.3f}")
    print(f"Ended at:   x={history[-1, 0]:.3f}, y={history[-1, 1]:.3f}")

    # --- VISUALIZATION PHASE ---
    fig = plt.figure(figsize=(14, 6))

    # Create a dynamic grid of points for the surface and contour maps
    # We size the grid based on the area the algorithm actually explored
    padding = 1.0
    x_min, x_max = np.min(history[:, 0]) - padding, np.max(history[:, 0]) + padding
    y_min, y_max = np.min(history[:, 1]) - padding, np.max(history[:, 1]) + padding

    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(X, Y)
    Z = f_num(X, Y)

    # Plot 1: 2D Contour Map (Top-down view)
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X, Y, Z, levels=40, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)

    # Plot the path taken
    ax1.plot(history[:, 0], history[:, 1], color='red', marker='.', markersize=6, label='Descent Path')
    ax1.plot(history[0, 0], history[0, 1], color='cyan', marker='*', markersize=14, label='Start')
    ax1.plot(history[-1, 0], history[-1, 1], color='magenta', marker='X', markersize=12, label='End (Min)')

    ax1.set_title("Top-Down Contour View")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    # Plot 2: 3D Surface Plot
    ax2 = fig.add_subplot(122, projection='3d')
    # Plot the mathematical surface
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    # Plot the 3D path over the surface
    ax2.plot(history[:, 0], history[:, 1], history[:, 2], color='red', marker='o', markersize=4, linewidth=2)
    ax2.scatter(history[0, 0], history[0, 1], history[0, 2], color='cyan', s=150, marker='*', zorder=5)
    ax2.scatter(history[-1, 0], history[-1, 1], history[-1, 2], color='magenta', s=100, marker='X', zorder=5)

    ax2.set_title(f"3D Surface View: f(x,y) = {func_expr}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x, y)")

    plt.tight_layout()
    plt.show()


# ==========================================
# Run Examples
# ==========================================
if __name__ == "__main__":
    visualize_gradient_descent(
        func_expr="x**2 + y**2",
        start_point=(4.0, 3.0),
        learning_rate=0.1,
        iterations=40
    )



    # Try your own! e.g., "sin(x) + cos(y) + x**2/10 + y**2/10"