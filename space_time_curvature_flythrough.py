import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(4)

N_FRAMES = 180
FPS = 15

CAMERA_SPEED = 0.18
PATH_AMPLITUDE_Y = 1.2
PATH_AMPLITUDE_Z = 0.7

WINDOW = 4.5
GRID_SPACING = 1.4
LINE_POINTS = 140

SOFTENING = 0.8
WARP_SCALE = 3.2
MAX_PULL = 0.7

types = [
    {"name": "blue_star",   "color": "#7ec8ff", "glow": "#4fa3ff", "mass": (1.5, 3.0)},
    {"name": "sun_like",    "color": "#ffd27f", "glow": "#ffb347", "mass": (1.0, 2.0)},
    {"name": "red_giant",   "color": "#ff6f61", "glow": "#ff3b2f", "mass": (2.0, 4.5)},
    {"name": "white_dwarf", "color": "#e6f2ff", "glow": "#cfd8ff", "mass": (0.8, 1.5)},
    {"name": "neutron",     "color": "#a88cff", "glow": "#7a5cff", "mass": (3.5, 6.0)},
]

N_OBJECTS = 16

obj_x = np.random.uniform(-2, 36, N_OBJECTS)
obj_y = np.random.uniform(-7, 7, N_OBJECTS)
obj_z = np.random.uniform(-6, 6, N_OBJECTS)

obj_type = np.random.choice(len(types), N_OBJECTS)

obj_mass = np.zeros(N_OBJECTS)
obj_size = np.zeros(N_OBJECTS)
obj_color = []
obj_glow = []

for i in range(N_OBJECTS):
    t = types[obj_type[i]]
    m = np.random.uniform(*t["mass"])
    obj_mass[i] = m
    obj_size[i] = 20 + 90 * (m / 6.0) ** 1.2
    obj_color.append(t["color"])
    obj_glow.append(t["glow"])

# BACKGROUND STARS
N_BG = 100
bg_x = np.random.uniform(-5, 40, N_BG)
bg_y = np.random.uniform(-10, 10, N_BG)
bg_z = np.random.uniform(-10, 10, N_BG)
bg_s = np.random.uniform(2, 8, N_BG)

# CAMERA
def camera_position(frame):
    x = frame * CAMERA_SPEED
    y = PATH_AMPLITUDE_Y * np.sin(0.06 * frame)
    z = PATH_AMPLITUDE_Z * np.sin(0.09 * frame)
    return np.array([x, y, z], dtype=float)

def camera_view(frame):
    elev = 15 + 7 * np.sin(0.03 * frame)
    azim = -65 + 6 * np.sin(0.04 * frame)
    return elev, azim


# GRID

def make_grid(cam):
    coords = np.arange(-WINDOW, WINDOW + 1e-9, GRID_SPACING)
    t = np.linspace(-WINDOW, WINDOW, LINE_POINTS)

    lines = []

    # x-directed lines
    for y0 in coords:
        for z0 in coords:
            x = cam[0] + t
            y = np.full_like(t, cam[1] + y0)
            z = np.full_like(t, cam[2] + z0)
            lines.append((x, y, z, 'x'))

    # y-directed lines
    for x0 in coords:
        for z0 in coords:
            x = np.full_like(t, cam[0] + x0)
            y = cam[1] + t
            z = np.full_like(t, cam[2] + z0)
            lines.append((x, y, z, 'y'))

    # z-directed lines
    for x0 in coords:
        for y0 in coords:
            x = np.full_like(t, cam[0] + x0)
            y = np.full_like(t, cam[1] + y0)
            z = cam[2] + t
            lines.append((x, y, z, 'z'))

    return lines

def warp(x, y, z):
    xw = np.array(x, dtype=float, copy=True)
    yw = np.array(y, dtype=float, copy=True)
    zw = np.array(z, dtype=float, copy=True)

    for ox, oy, oz, m in zip(obj_x, obj_y, obj_z, obj_mass):
        dx = xw - ox
        dy = yw - oy
        dz = zw - oz

        r2 = dx**2 + dy**2 + dz**2 + SOFTENING**2
        r = np.sqrt(r2)

        pull = np.minimum(WARP_SCALE * m / r2, MAX_PULL)

        xw = xw - pull * dx / r
        yw = yw - pull * dy / r
        zw = zw - pull * dz / r

    return xw, yw, zw

fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

def update(frame):
    ax.clear()
    ax.set_facecolor('black')
    ax.set_axis_off()

    cam = camera_position(frame)
    elev, azim = camera_view(frame)

    # background stars
    mask = (
        (bg_x > cam[0] - 3) & (bg_x < cam[0] + 10) &
        (np.abs(bg_y - cam[1]) < 10) &
        (np.abs(bg_z - cam[2]) < 10)
    )
    ax.scatter(
        bg_x[mask], bg_y[mask], bg_z[mask],
        s=bg_s[mask], c='white', alpha=0.35, linewidths=0
    )

    # celestial objects near camera
    visible = (
        (obj_x > cam[0] - 3) & (obj_x < cam[0] + 10) &
        (np.abs(obj_y - cam[1]) < 8) &
        (np.abs(obj_z - cam[2]) < 8)
    )

    for i in np.where(visible)[0]:
        ox, oy, oz = obj_x[i], obj_y[i], obj_z[i]

        ax.scatter([ox], [oy], [oz], s=obj_size[i] * 6,
                   c=obj_glow[i], alpha=0.05, edgecolors='none')
        ax.scatter([ox], [oy], [oz], s=obj_size[i] * 2,
                   c=obj_glow[i], alpha=0.12, edgecolors='none')
        ax.scatter([ox], [oy], [oz], s=obj_size[i],
                   c=obj_color[i], alpha=0.95, edgecolors='none')

    # curved grid
    lines = make_grid(cam)

    for x, y, z, fam in lines:
        xw, yw, zw = warp(x, y, z)

        if fam == 'x':
            color = (0.4, 0.7, 1.0)
        elif fam == 'y':
            color = (0.8, 0.4, 1.0)
        else:
            color = (0.3, 1.0, 0.8)

        ax.plot(xw, yw, zw, color=color, alpha=0.25, linewidth=0.7)

    ax.set_xlim(cam[0] - WINDOW, cam[0] + WINDOW)
    ax.set_ylim(cam[1] - WINDOW, cam[1] + WINDOW)
    ax.set_zlim(cam[2] - WINDOW, cam[2] + WINDOW)

    ax.view_init(elev=elev, azim=azim)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS)
#ani.save("spacetime_celestial_flythrough.gif", writer=PillowWriter(fps=FPS))

print("Saved as spacetime_celestial_flythrough.gif")
