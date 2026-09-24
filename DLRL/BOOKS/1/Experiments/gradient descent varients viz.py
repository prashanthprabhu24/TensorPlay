import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ==========================================
# 1. Define the Landscape
# ==========================================
# We use an "Elongated Bowl" (f(x,y) = x^2 + 10y^2).
# This creates a canyon shape. Vanilla GD bounces wildly across the canyon
# walls, while Momentum and Adam slice right down the middle.
def f(x, y):
    return x ** 2 + 10 * y ** 2


def grad_f(x, y):
    return np.array([2 * x, 20 * y])


# ==========================================
# 2. Define the Optimizers
# ==========================================
class Optimizer:
    def __init__(self, name, color, lr, start_pos):
        self.name = name
        self.color = color
        self.lr = lr
        self.pos = np.array(start_pos, dtype=float)
        self.path = [self.pos.copy()]


class VanillaGD(Optimizer):
    def step(self, grad):
        self.pos -= self.lr * grad
        self.path.append(self.pos.copy())


class Momentum(Optimizer):
    def __init__(self, name, color, lr, start_pos, beta=0.9):
        super().__init__(name, color, lr, start_pos)
        self.beta = beta
        self.velocity = np.zeros(2)

    def step(self, grad):
        self.velocity = self.beta * self.velocity + (1 - self.beta) * grad
        self.pos -= self.lr * self.velocity
        self.path.append(self.pos.copy())


class Adam(Optimizer):
    def __init__(self, name, color, lr, start_pos, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(name, color, lr, start_pos)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # Bias corrections
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        self.pos -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.path.append(self.pos.copy())


# ==========================================
# 3. Setup and Pre-calculate Paths
# ==========================================
start_position = [-8.0, 3.0]
iterations = 100

# Initialize our racers. (Learning rates are tuned so they are comparable)
optimizers = [
    VanillaGD("Vanilla GD", "gray", lr=0.085, start_pos=start_position),
    Momentum("Momentum", "blue", lr=0.085, start_pos=start_position),
    Adam("Adam", "red", lr=0.5, start_pos=start_position)  # Adam needs a higher LR here
]

# Run the simulation for all optimizers
for i in range(iterations):
    for opt in optimizers:
        g = grad_f(opt.pos[0], opt.pos[1])
        opt.step(g)

# Convert paths to numpy arrays for easier plotting
for opt in optimizers:
    opt.path = np.array(opt.path)

# ==========================================
# 4. Visualization & Animation
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the background surface
x_grid = np.linspace(-10, 10, 50)
y_grid = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.4, edgecolor='none')

# Setup empty lines and points for the animation
lines = []
points = []
for opt in optimizers:
    # The trailing line
    line, = ax.plot([], [], [], color=opt.color, label=opt.name, linewidth=2)
    lines.append(line)
    # The leading point (the "ball")
    point, = ax.plot([], [], [], color=opt.color, marker='o', markersize=8)
    points.append(point)

ax.set_title("Optimizer Race: Navigating the Canyon", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(X, Y)")
ax.view_init(elev=35, azim=-120)  # Set a good starting camera angle
ax.legend()


# Animation update function
def update(frame):
    for i, opt in enumerate(optimizers):
        # Get data up to the current frame
        x_data = opt.path[:frame + 1, 0]
        y_data = opt.path[:frame + 1, 1]
        z_data = f(x_data, y_data)

        # Update line
        lines[i].set_data(x_data, y_data)
        lines[i].set_3d_properties(z_data)

        # Update the leading point
        points[i].set_data([x_data[-1]], [y_data[-1]])
        points[i].set_3d_properties([z_data[-1]])

    return lines + points


# Create the animation object
# interval = milliseconds between frames
ani = FuncAnimation(fig, update, frames=iterations, interval=50, blit=False)

# plt.show()

# NOTE: If you want to save it as an mp4 video file, uncomment the lines below:
# (Requires FFmpeg installed on your system)
print("Saving video...")
ani.save("optimizer_race.mp4", writer='ffmpeg', fps=20)
print("Done!")