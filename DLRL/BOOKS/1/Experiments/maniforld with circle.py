import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

radius = 5
center_x = 0
center_y = 0
theta = np.linspace(0, 2 * np.pi, 1000)
x = center_x + radius * np.cos(theta)
y = center_y + radius * np.sin(theta)
plt.plot(x, y)
plt.axis('equal')
plt.title("Circle")
plt.show()


