import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib import cm
matplotlib.use("Qt5Agg")
import numpy as np

# Parameters
num_stages = 5  # Number of stages
lengths = [1.5, 1.3, 1.1, 0.9, 0.7]  # Lengths of the pendulums
amplitudes = [1.0, 0.6, 0.3, 0.1, 0.05]  # Initial amplitudes of vibration
frequencies = [0.5, 0.4, 0.3, 0.2, 0.1]  # Natural frequencies of each stage
damping_factor = 0.1  # Exponential decay
time = np.linspace(0, 60, 800)  # Time for the animation

# Initialize plot
#fig, ax = plt.subplots()
fig, (ax, resonance_ax) = plt.subplots(1, 2, figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[3, 1])  # Adjust width ratios to make the subplot smaller

#ax = fig.add_subplot(gs[0])
#resonance_ax = fig.add_subplot(gs[1])
ax.set_xlim(-2, 2)
ax.set_ylim(-1, sum(lengths) + 1)

# Add subplot for resonance visualization
#resonance_ax = fig.add_axes([0.65, 0.5, 0.25, 0.5])  # Small inset subplot #1.05, 0.1, 0.25, 0.8
resonance_ax.set_xlim(0, time[-1])
resonance_ax.set_ylim(0, 1)
#resonance_ax.set_title("amplitudes")
resonance_ax.set_xlabel("Time (s)")
resonance_ax.set_ylabel("Amplitude")

# Elements for animation
lines = []
masses = []
resonance_lines = []
mass_size = [7, 7, 7, 7, 13]
mass_style = ['*', '*','*','*','o']
blues = cm.get_cmap("Blues", 10)
print(blues)
mass_color = [blues(6), blues(7), blues(8), blues(9),  '#d62728']
wei = [1, 2, 3, 4, 5]
# Create lines and masses for each stage
for i in range(num_stages):
    line, = ax.plot([], [], 'k-', lw=2)
    mass, = ax.plot([], [], color = mass_color[i], marker = mass_style[i], markersize=mass_size[i])
    res_line, = resonance_ax.plot([], [], color = mass_color[i], label=f"Stage {i+1}")
    lines.append(line)
    masses.append(mass)
    resonance_lines.append(res_line)

# Add legend to the resonance plot
resonance_ax.legend()

# Initialize function
def init():
    for line, mass, res_line in zip(lines, masses, resonance_lines):
        line.set_data([], [])
        mass.set_data([], [])
        res_line.set_data([], [])
    return lines + masses + resonance_lines

# Animation function
def animate(t):
    x_prev, y_prev = 0, sum(lengths)
    positions = []  # To store each stage's end positions
    resonance_data = [[] for _ in range(num_stages)]  # For resonance plot

    for i in range(num_stages):
        # Compute displacement with resonance and damping
        x_displacement = (
            amplitudes[i]
            * np.exp(-damping_factor * t)
            * np.sin(2 * np.pi * frequencies[i] * t)
        )
        x_curr, y_curr = x_prev + x_displacement, y_prev - lengths[i]
        positions.append((x_curr, y_curr))

        # Update lines and masses
        lines[i].set_data([x_prev, x_curr], [y_prev, y_curr])
        masses[i].set_data(x_curr, y_curr)

        # Store resonance data
        resonance_data[i].append(abs(x_displacement))
        x_prev, y_prev = x_curr, y_curr

    # Update resonance subplot
    for i, res_line in enumerate(resonance_lines):
        res_line.set_data(time[:int(t * 20)], resonance_data[i])

    return lines + masses + resonance_lines

# Create animation
ani = FuncAnimation(fig, animate, frames=time, init_func=init, blit=False, interval=60)

plt.suptitle("Super Attenuator with multiple stages")
plt.show()
# ani.save('animation.gif', writer='pillow')
