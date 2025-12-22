# import numpy as np
# import matplotlib
# matplotlib.use("Qt5Agg")
#
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # ==========================================================
# # Planetary data: (name, semi-major axis [AU], period [yr], color, size)
# # ==========================================================
# planets = [
#     ("Mercury", 0.39, 0.241, "gray",   6),
#     ("Venus",   0.72, 0.615, "orange", 8),
#     ("Earth",   1.00, 1.000, "blue",   10),
#     ("Mars",    1.52, 1.881, "red",    8),
#     ("Jupiter", 5.20, 11.86, "orange", 14),
#     ("Saturn",  9.58, 29.46, "gold",   12),
#     ("Uranus", 19.20, 84.01, "cyan",   11),
#     ("Neptune",30.05,164.8,  "blue",   11),
#     ("Pluto",  39.48,248.0,  "white",  6),
# ]
#
# # ==========================================================
# # Scaling (logarithmic for visibility)
# # ==========================================================
# def scale_r(a):
#     return np.log10(1 + a)
#
# # ==========================================================
# # Figure
# # ==========================================================
# fig, ax = plt.subplots(figsize=(9, 9))
# fig.patch.set_facecolor("black")
# ax.set_facecolor("black")
# ax.set_aspect("equal")
# ax.set_xticks([])
# ax.set_yticks([])
#
# # ==========================================================
# # Sun
# # ==========================================================
# sun = ax.scatter(0, 0, s=350, color="yellow", zorder=10)
#
# # ==========================================================
# # Orbits & planets
# # ==========================================================
# planet_dots = []
# orbit_lines = []
# angles = []
#
# for name, a, T, color, size in planets:
#     r = scale_r(a)
#     th = np.linspace(0, 2*np.pi, 400)
#     orbit, = ax.plot(r*np.cos(th), r*np.sin(th),
#                      color="white", lw=0.6, alpha=0.3)
#     orbit_lines.append(orbit)
#
#     dot = ax.scatter([], [], s=size**2, color=color, zorder=5)
#     planet_dots.append(dot)
#
#     angles.append(np.random.uniform(0, 2*np.pi))
#
# # ==========================================================
# # Time control
# # ==========================================================
# dt = 0.02  # years per frame
#
# # Zoom control
# zoom_start = 150      # frame where zoom begins
# zoom_end = 320        # frame where zoom ends
#
# full_lim = scale_r(42)
# earth_lim = scale_r(1.3)
#
# # ==========================================================
# # Animation
# # ==========================================================
# def update(frame):
#     # --- Planet motion ---
#     for i, (name, a, T, color, size) in enumerate(planets):
#         omega = 2*np.pi / T
#         angles[i] += omega * dt
#         r = scale_r(a)
#         x = r * np.cos(angles[i])
#         y = r * np.sin(angles[i])
#         planet_dots[i].set_offsets([x, y])
#
#     # --- Smooth zoom ---
#     if frame < zoom_start:
#         lim = full_lim
#     elif frame > zoom_end:
#         lim = earth_lim
#     else:
#         f = (frame - zoom_start) / (zoom_end - zoom_start)
#         # Smoothstep interpolation
#         f = f*f*(3 - 2*f)
#         lim = full_lim*(1 - f) + earth_lim*f
#
#     ax.set_xlim(-lim, lim)
#     ax.set_ylim(-lim, lim)
#
#     return planet_dots
#
# # Initial limits
# ax.set_xlim(-full_lim, full_lim)
# ax.set_ylim(-full_lim, full_lim)
#
# plt.title("Solar System → Earth (Smooth Zoom)", color="white", pad=20)
#
# ani = FuncAnimation(fig, update, interval=40, blit=False)
# plt.show()


import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ==========================================================
# Planetary data
# ==========================================================
planets = [
    ("Mercury", 0.39, 0.241, "gray",   6),
    ("Venus",   0.72, 0.615, "orange", 8),
    ("Earth",   1.00, 1.000, "blue",   10),
    ("Mars",    1.52, 1.881, "red",    8),
    ("Jupiter", 5.20, 11.86, "orange", 14),
    ("Saturn",  9.58, 29.46, "gold",   12),
    ("Uranus", 19.20, 84.01, "cyan",   11),
    ("Neptune",30.05,164.8,  "blue",   11),
    ("Pluto",  39.48,248.0,  "white",  6),
]

def scale_r(a):
    return np.log10(1 + a)

earth_index = [p[0] for p in planets].index("Earth")

# ==========================================================
# Figure
# ==========================================================
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

# ==========================================================
# Sun
# ==========================================================
sun = ax.scatter(0, 0, s=350, color="yellow", zorder=10)

# ==========================================================
# Orbits & planets
# ==========================================================
planet_dots = []
orbit_lines = []
angles = []

for name, a, T, color, size in planets:
    r = scale_r(a)
    th = np.linspace(0, 2*np.pi, 400)
    orbit, = ax.plot(r*np.cos(th), r*np.sin(th),
                     color="white", lw=0.6, alpha=0.3)
    orbit_lines.append(orbit)

    dot = ax.scatter([], [], s=size**2, color=color, zorder=5)
    planet_dots.append(dot)

    angles.append(np.random.uniform(0, 2*np.pi))

# ==========================================================
# Earth sphere (initially hidden)
# ==========================================================
earth_sphere = Circle((0, 0), 0.0,
                      facecolor="#1f77b4",
                      edgecolor="white",
                      lw=1.2,
                      zorder=30,
                      alpha=0.0)
ax.add_patch(earth_sphere)

shade = Circle((0, 0), 0.0,
               facecolor="black",
               alpha=0.25,
               zorder=31)
ax.add_patch(shade)

# ==========================================================
# Timing
# ==========================================================
dt = 0.02

zoom1_start, zoom1_end = 120, 280
zoom2_start, zoom2_end = 300, 420

full_lim = scale_r(42)
earth_orb_lim = scale_r(1.3)
earth_zoom_lim = 0.15

# ==========================================================
# Animation
# ==========================================================
def update(frame):
    # --- Planet motion ---
    for i, (name, a, T, color, size) in enumerate(planets):
        omega = 2*np.pi / T
        angles[i] += omega * dt
        r = scale_r(a)
        x = r * np.cos(angles[i])
        y = r * np.sin(angles[i])
        planet_dots[i].set_offsets([x, y])

    # Earth's current position
    ex, ey = planet_dots[earth_index].get_offsets()[0]

    # --- Zoom logic ---
    if frame < zoom1_start:
        lim = full_lim
        cx, cy = 0, 0

    elif frame < zoom1_end:
        f = (frame - zoom1_start) / (zoom1_end - zoom1_start)
        f = f*f*(3 - 2*f)
        lim = full_lim*(1 - f) + earth_orb_lim*f
        cx, cy = 0, 0

    elif frame < zoom2_start:
        lim = earth_orb_lim
        cx, cy = 0, 0

    elif frame < zoom2_end:
        f = (frame - zoom2_start) / (zoom2_end - zoom2_start)
        f = f*f*(3 - 2*f)

        # Move camera center from Sun → Earth
        cx = f * ex
        cy = f * ey
        lim = earth_orb_lim*(1 - f) + earth_zoom_lim*f

        # Fade out Sun
        sun.set_alpha(1 - f)

        # Fade out orbits
        for orb in orbit_lines:
            orb.set_alpha(0.3*(1 - f))

        # Grow Earth sphere at Earth's position
        earth_sphere.center = (ex, ey)
        earth_sphere.radius = 0.08 * f
        earth_sphere.set_alpha(f)

        shade.center = (ex + 0.02*f, ey - 0.02*f)
        shade.radius = earth_sphere.radius

    else:
        cx, cy = ex, ey
        lim = earth_zoom_lim

    ax.set_xlim(cx - lim, cx + lim)
    ax.set_ylim(cy - lim, cy + lim)

    return planet_dots

ax.set_xlim(-full_lim, full_lim)
ax.set_ylim(-full_lim, full_lim)

plt.title("Solar System → Earth (Correct Camera Lock)",
          color="white", pad=20)

ani = FuncAnimation(fig, update, interval=40, blit=False)
plt.show()


