import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import cm

# --- Parameters ---
num_stages = 5  # Total number of masses (stage 1 = test mass, then stages 2-5)
# For the chain, we define rod lengths from one mass to the next (for stages 2 to 5):
rod_lengths = [1.5, 1.3, 1.1, 0.9]  # There will be 4 rods
# For the oscillatory behavior, we need 5 values (one for each mass)
amplitudes = [0.5, 0.4, 0.3, 0.2, 0.1]
frequencies = [0.5, 0.4, 0.3, 0.2, 0.1]
damping_factor = 0.1

total_time = 50  # seconds
frames = 1000
time_array = np.linspace(0, total_time, frames)

# --- Set up the plot ---
# We choose y = 0 for the test mass, and the chain will build upward.
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
# The maximum y is the sum of rod lengths plus a little margin.
ax.set_ylim(-0.5, sum(rod_lengths) + 1)
ax.set_title("Multi Stage Suspension")
ax.set_xlabel("Horizontal Displacement (m)")
ax.set_ylabel("Vertical Position (m)")

# Create lists to hold line (rod) and mass (marker) objects for all stages.
# Note: For stage 1 (test mass) we only draw a mass (no rod from below).
lines = []  # rods connecting masses (stages 2 and above)
masses = []  # markers for each mass
mass_styles = ['ro','b*','b*','b*','b*' ]
mass_sizes = [13, 7, 7, 7, 7]
for i in range(num_stages):
    line, = ax.plot([], [], 'k-', lw=2)
    mass, = ax.plot([], [], mass_styles[i], markersize=mass_sizes[i])
    lines.append(line)
    masses.append(mass)

# Text to display test mass horizontal displacement
disp_text = ax.text(-2.3, 0.2, "", fontsize=10, color="red")
label_colors = ["red", 'blue',  'blue', 'blue', 'blue']
# Create text objects for stage labels (one per stage)
stage_labels = []
for j in range(num_stages):
    # Initialize with empty text at (0,0); positions will be updated later.
    label = ax.text(0, 0, "", fontsize=8, color=label_colors[j])
    stage_labels.append(label)

# We start by drawing only the test mass.
stages_to_draw = 1


# --- Initialization Function ---
def init():
    for line, mass, label in zip(lines, masses, stage_labels):
        line.set_data([], [])
        mass.set_data([], [])
        label.set_text("")
    disp_text.set_text("")
    return lines + masses + [disp_text] + stage_labels


# --- Animation Function ---
def animate(frame):
    global stages_to_draw
    t = time_array[frame]

    # Every ~100 frames (~5 seconds), add one more stage until the full chain is built.
    if frame % 250 == 0 and stages_to_draw < num_stages:
        stages_to_draw += 1

    # Compute horizontal displacements for each stage currently drawn.
    # For stage i (0-indexed), define: d_i = amplitude[i] * exp(-damping*t) * sin(2π*frequency[i]*t)
    displacements = [amplitudes[i] * np.exp(-damping_factor * t) * np.sin(2 * np.pi * frequencies[i] * t)
                     for i in range(stages_to_draw)]

    # Determine positions:
    # For stage 1 (test mass), equilibrium is at (0, 0); its actual position is (d_0, 0)
    positions = []
    x0 = displacements[0]
    y0 = 0
    positions.append((x0, y0))

    # For each subsequent stage i (i = 2,..., stages_to_draw),
    # the equilibrium vertical position is the cumulative sum of rod lengths.
    # The horizontal position is the cumulative sum of displacements.
    for i in range(1, stages_to_draw):
        y_eq = sum(rod_lengths[:i])  # equilibrium y for stage i+1
        x_val = sum(displacements[:i + 1])
        positions.append((x_val, y_eq))

    # Update drawing:

    # For Stage 1, only update its mass marker.
    masses[0].set_data(positions[0][0], positions[0][1])
    # Place its stage label to the left and slightly above its marker.
    stage_labels[0].set_text("mirror")
    stage_labels[0].set_position((positions[0][0] - 0.4, positions[0][1] + 0.2))
    # For stage 1, draw only the mass.
    masses[0].set_data(positions[0][0], positions[0][1])

    # For stages 2 and above, draw a rod from the previous mass to the current mass, and draw the mass.
    for i in range(1, stages_to_draw):
        x_prev, y_prev = positions[i - 1]
        x_curr, y_curr = positions[i]
        lines[i].set_data([x_prev, x_curr], [y_prev, y_curr])
        masses[i].set_data(x_curr, y_curr)
        stage_labels[i].set_text(f"S{i}")
        stage_labels[i].set_position((x_curr - 0.4, y_curr + 0.2))

    # Clear objects for stages not yet drawn.
    for i in range(stages_to_draw, num_stages):
        lines[i].set_data([], [])
        masses[i].set_data([], [])
        stage_labels[i].set_text("")

    # Display the test mass displacement (its horizontal coordinate).
    disp_text.set_text(f"mirror motion: \n {positions[0][0]:.3f} m")

    return lines + masses + [disp_text] +stage_labels


ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False, interval=25)
plt.show()
#ani.save('animation_pub.gif', writer='pillow')

