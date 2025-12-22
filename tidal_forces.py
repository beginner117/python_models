import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
# ---- Force an interactive backend (important in many IDEs) ----
# Try QtAgg first, then TkAgg.
for bk in ("QtAgg", "TkAgg"):
    try:
        matplotlib.use(bk, force=True)
        break
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Physical constants (SI)
# -----------------------
G = 6.67430e-11

# Earth
R_E = 6.371e6                 # m
omega_E = 2 * np.pi / 86164   # rad/s (sidereal day)

# Moon
M_m = 7.342e22
D_m = 3.844e8
T_m = 27.321661 * 86400
omega_m = 2 * np.pi / T_m

# Sun
M_s = 1.9885e30
D_s = 1.496e11
T_s = 365.256363004 * 86400
omega_s = 2 * np.pi / T_s

# -----------------------
# Simulation controls
# -----------------------
N = 360
theta_earth = np.linspace(0, 2*np.pi, N, endpoint=False)  # fixed points on Earth's surface (Earth frame)

dt = 1200.0  # simulated seconds per frame (20 minutes)
visual_bulge_scale = 1.2e5  # purely visual scale factor
response_relax = 0.20       # ocean "response" smoothing 0..1

# -----------------------
# Helpers
# -----------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s],
                     [s,  c]])

def tidal_potential(M, D, cos_gamma, R=R_E):
    """
    Leading-order tidal potential at Earth's surface (J/kg):
    U_tide = - (G M / (2 D^3)) * R^2 * (3 cos^2(gamma) - 1)
    """
    return - (G * M / (2.0 * D**3)) * (R**2) * (3.0 * cos_gamma**2 - 1.0)

def body_dir(t, omega, phase0=0.0):
    """Unit direction from Earth to body in inertial frame (2D)."""
    ang = omega * t + phase0
    return np.array([np.cos(ang), np.sin(ang)])

# -----------------------
# Figure setup
# -----------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.set_aspect("equal", adjustable="box")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")

margin = 1.7 * R_E
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_xticks([])
ax.set_yticks([])

# Earth disk
earth = plt.Circle((0, 0), R_E, color="#1f77b4", alpha=0.95)
ax.add_patch(earth)

# A simple "continent-like" blob drawn on Earth, rotated each frame (visual cue)
blob_t = np.linspace(0, 2*np.pi, 300)
blob_r = 0.55 * R_E * (1.0 + 0.15*np.sin(3*blob_t) + 0.08*np.sin(7*blob_t + 0.9))
blob_x0 = blob_r * np.cos(blob_t)
blob_y0 = blob_r * np.sin(blob_t)
#blob_line, = ax.plot([], [], lw=2.0, alpha=0.65, color="red")  # green

# Rotating meridian line (another rotation cue)
meridian, = ax.plot([], [], lw=1.5, alpha=0.7, color="white")

# Ocean reference and bulge ring
ocean_ref,  = ax.plot([], [], lw=1.2, alpha=0.25, color="white")
ocean_line, = ax.plot([], [], lw=2.8, color="#00e5ff")

# Moon/Sun direction markers (placed near edge for visibility)
moon_scatter = ax.scatter([], [], s=90, color="#cfcfcf")
sun_scatter  = ax.scatter([], [], s=130, color="#ffd166")
moon_dir_line, = ax.plot([], [], lw=1.2, alpha=0.65, color="#cfcfcf")
sun_dir_line,  = ax.plot([], [], lw=1.2, alpha=0.65, color="#ffd166")

info_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    va="top", ha="left", fontsize=11, color="white",
    family="monospace"
)

plt.title("Newtonian Ocean Tides (Moon + Sun) — Earth rotates, bulges move", color="white", pad=14)

prev_bulge = np.zeros(N)

def init():
    # Reference circle in Earth frame
    ref_x = R_E * np.cos(theta_earth)
    ref_y = R_E * np.sin(theta_earth)
    ocean_ref.set_data(ref_x, ref_y)
    ocean_line.set_data(ref_x, ref_y)

    # Meridian initial
    meridian.set_data([0, 0], [-R_E, R_E])

    return (meridian, ocean_ref, ocean_line,
            moon_scatter, sun_scatter, moon_dir_line, sun_dir_line, info_text)

def update(frame):
    global prev_bulge
    t = frame * dt

    # Earth rotation angle
    phi = omega_E * t

    # ---- Rotate Earth features for visible rotation ----
    Rphi = rot2(phi)
    blob_xy = Rphi @ np.vstack([blob_x0, blob_y0])
    #blob_line.set_data(blob_xy[0], blob_xy[1])

    # Meridian (a radius line) rotated with Earth
    m_end = Rphi @ np.array([0.0, R_E])
    meridian.set_data([0, m_end[0]], [0, m_end[1]])

    # ---- Compute tide forcing in the EARTH FRAME ----
    # Moon/Sun directions in inertial frame:
    m_dir_in = body_dir(t, omega_m, phase0=0.0)
    s_dir_in = body_dir(t, omega_s, phase0=np.pi/3)

    # Convert to Earth frame by undoing Earth rotation:
    Rmphi = rot2(-phi)
    m_dir = unit(Rmphi @ m_dir_in)
    s_dir = unit(Rmphi @ s_dir_in)

    # Surface unit vectors in Earth frame (fixed grid on Earth):
    rhat = np.stack([np.cos(theta_earth), np.sin(theta_earth)], axis=1)  # (N,2)

    # cos(gamma) = rhat · dir_body (both in Earth frame)
    cos_g_m = rhat @ m_dir
    cos_g_s = rhat @ s_dir

    U_m = tidal_potential(M_m, D_m, cos_g_m)
    U_s = tidal_potential(M_s, D_s, cos_g_s)

    bulge_target = visual_bulge_scale * (-(U_m + U_s))
    bulge_target -= np.mean(bulge_target)  # keep average radius fixed

    # Smooth response
    bulge = (1 - response_relax) * prev_bulge + response_relax * bulge_target
    prev_bulge = bulge

    # Ocean ring in Earth frame (bulges move over fixed surface points)
    R = R_E + bulge
    x = R * np.cos(theta_earth)
    y = R * np.sin(theta_earth)
    ocean_line.set_data(x, y)

    # ---- Show Moon/Sun directions (in inertial frame, but draw relative to Earth) ----
    # For drawing in the plot (Earth frame), we can just use m_dir and s_dir directly.
    # moon_dir_line.set_data([0, 1.25*R_E*m_dir[0]], [0, 1.25*R_E*m_dir[1]])
    # sun_dir_line.set_data([0, 1.25*R_E*s_dir[0]], [0, 1.25*R_E*s_dir[1]])

    # moon_vis = 1.50 * R_E * m_dir
    # sun_vis  = 1.50 * R_E * s_dir
    # moon_scatter.set_offsets([moon_vis[0], moon_vis[1]])
    # sun_scatter.set_offsets([sun_vis[0], sun_vis[1]])

    # Diagnostics
    moon_strength = M_m / (D_m**3)
    sun_strength  = M_s / (D_s**3)
    ratio = moon_strength / sun_strength
    alignment = float(np.clip(np.dot(m_dir, s_dir), -1, 1))
    sim_days = t / 86400.0

    info_text.set_text(
        "Tidal bulges from differential gravity\n"
        f"Sim time: {sim_days:7.2f} days\n"
        f"Earth rotation: {np.degrees(phi)%360:6.1f}°\n"
        f"Moon tidal strength / Sun: {ratio:5.2f}×\n"
        f"Moon–Sun alignment (cos): {alignment: .3f}\n"
        "\n"
        # "Bulges move over Earth as Earth rotates.\n"
        # "Alignment near ±1 => spring tides; near 0 => neap tides."
    )

    return (meridian, ocean_ref, ocean_line,
            moon_scatter, sun_scatter, moon_dir_line, sun_dir_line, info_text)


# IMPORTANT: keep a reference to ani (don't let it be garbage-collected)
ani = FuncAnimation(fig, update, init_func=init, interval=33, blit=False)

plt.show()
