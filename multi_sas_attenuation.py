import numpy as np
import matplotlib.pyplot as plt

"""
Passive isolation models for GW suspensions:
- Inverted Pendulum (IP): before vs. after percussion clearance (counter-mass tuning)
- GAS (Geometric Anti-Spring) vertical filters: 1, 2, and 3 cascaded stages
- Horizontal pendulum: 1 vs. 2 cascaded stages
- Illustrative full passive chain: IP (tuned) × 3×GAS × 2×Pendulum

All transfer functions are |X / X0|(f) from base motion to payload motion.
"""

# --- Frequency grid ---
f = np.logspace(-3, 2, 4000)
w = 2 * np.pi * f

# --------------------------
# Inverted Pendulum (IP)
# ---------------------------
def H_ip_basic(w, f0=0.1, beta=0.07, phi=1e-4):
    """
    Inverted Pendulum (before percussion clearance).

    Model (from your text):
        H(ω) = [ ω0^2 (1 + i φ) + β ω^2 ] / [ ω0^2 (1 + i φ) - ω^2 ]

    where:
        f0  : natural frequency (Hz), ω0 = 2π f0
        β   : percussion parameter (coupling of leg inertia -> flattening at high f)
        φ   : small loss angle for internal dissipation

    Large β causes the high-frequency "flattening" (bad isolation).
    """
    w0 = 2 * np.pi * f0
    num = (w0**2) * (1 + 1j * phi) + beta * (w**2)
    den = (w0**2) * (1 + 1j * phi) - (w**2)
    return num / den

def H_ip_countermass(w, f0=0.1, gamma=0.01, phi=1e-4):
    """
    Inverted Pendulum after counter-mass tuning (percussion clearance).

    Same structure as H_ip_basic but with β -> γ, where γ << β
    due to counter-masses and lighter legs (percussion point closer to hinge).
    """
    w0 = 2 * np.pi * f0
    num = (w0**2) * (1 + 1j * phi) + gamma * (w**2)
    den = (w0**2) * (1 + 1j * phi) - (w**2)
    return num / den

# ---------------------------
# GAS vertical filters
# ---------------------------
def H_gas_single(w, f0=0.3, M=350.0, m=110.0, phi=1e-3, gamma=0.05):
    """
    Single GAS filter (vertical) TF from base to payload:

        H(ω) = [ ω0^2 (1 + i φ) + (m/M) ω^2 ] /
               [ ω0^2 (1 + i φ) - ω^2 + i (γ/M) ω ]

    Parameters:
        f0    : GAS resonance (Hz)
        M     : payload mass (kg)
        m     : effective blade/inertia mass (kg)
        φ     : internal loss angle
        gamma : viscous-like damping constant (scaled by M in the denominator)

    Notes:
        - The (m/M) term slightly spoils perfect low-f response (finite transmissibility).
        - Above resonance the magnitude rolls off.
    """
    w0 = 2 * np.pi * f0
    num = (w0**2) * (1 + 1j * phi) + (m / M) * (w**2)
    den = (w0**2) * (1 + 1j * phi) - (w**2) + 1j * (gamma / M) * w
    return num / den


def H_gas_cascade(w, f0_list, M=350.0, m=11.0, phi=1e-3, gamma=0.05):
    """
    Cascade multiple GAS filters (multiply transfer functions).
    f0_list : list of resonant frequencies for each stage (Hz).
    """
    H = np.ones_like(w, dtype=complex)
    for f0 in f0_list:
        H *= H_gas_single(w, f0=f0, M=M, m=m, phi=phi, gamma=gamma)
    return H


# ---------------------------
# Horizontal pendulum(s)
# ---------------------------
def H_pend_single(w, f0=0.5, Q=50.0):
    """
    Simple pendulum (horizontal) TF from base to payload:

        H(ω) = ω0^2 / ( ω0^2 - ω^2 + i ω ω0 / Q )

    Parameters:
        f0 : pendulum resonance (Hz) with ω0 = 2π f0
        Q  : quality factor (losses)

    Above resonance, |H| ~ (ω0/ω)^2 -> excellent horizontal isolation.
    """
    w0 = 2 * np.pi * f0
    num = w0**2
    den = (w0**2) - (w**2) + 1j * w * w0 / Q
    return num / den


def H_pend_cascade(w, f0_list, Q=50.0):
    """
    Cascade multiple horizontal pendulums (multiply transfer functions).
    f0_list : list of resonant frequencies (Hz).
    """
    H = np.ones_like(w, dtype=complex)
    for f0 in f0_list:
        H *= H_pend_single(w, f0=f0, Q=Q)
    return H

# --- Compute transfer functions ---
H_ip_before = H_ip_basic(w, f0=0.1, beta=0.07, phi=1e-4)
H_ip_after  = H_ip_countermass(w, f0=0.1, gamma=0.01, phi=1e-4)
H_gas_1 = H_gas_cascade(w, [0.2])
H_gas_2 = H_gas_cascade(w, [0.2, 0.3])
H_gas_3 = H_gas_cascade(w, [0.2, 0.3, 0.4])
H_p1 = H_pend_cascade(w, [0.3])
H_p2 = H_pend_cascade(w, [0.3, 0.5])
H_chain = H_ip_after *  H_p2 

# --- Plotting ---
plt.figure(figsize=(7, 5))
plt.loglog(f, np.abs(H_ip_before), label="IP (before counter mass)")
plt.loglog(f, np.abs(H_ip_after), label="IP (after counter-mass tuning)")
plt.xlabel("Frequency [Hz]", fontsize=13, fontname='Arial')
plt.ylabel(r"$|X/X_0|$", fontsize=13, fontname='Arial')
plt.title("Inverted Pendulum Transfer Function \n Before vs After Counter Mass Tuning", fontsize=14, fontname='Arial')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()


plt.figure(figsize=(7, 5))
plt.loglog(f, np.abs(H_gas_1), label="1 stage (f0 = 0.2 Hz)")
plt.loglog(f, np.abs(H_gas_2), label="2 stages (0.2, 0.3 Hz)")
plt.loglog(f, np.abs(H_gas_3), label="3 stages (0.2, 0.3, 0.4 Hz)")
plt.xlabel("Frequency [Hz]", fontsize=13, fontname='Arial')
plt.ylabel(r"$|X/X_0|$", fontsize=13, fontname='Arial')
plt.title("GAS filter (Vertical Isolation) transfer function", fontsize=14, fontname='Arial')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(7, 5))
plt.loglog(f, np.abs(H_p1), label="1 stage (0.3 Hz)")
plt.loglog(f, np.abs(H_p2), label="2 stages (0.3, 0.5 Hz)")
plt.xlabel("Frequency [Hz]", fontsize=13, fontname='Arial')
plt.ylabel(r"$|X/X_0|$", fontsize=13, fontname='Arial')
plt.title("Horizontal Pendulum transfer function", fontsize=14, fontname='Arial')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(7, 5))
plt.loglog(f, np.abs(H_chain), label="Full Horizontal isolation ")
plt.loglog(f, np.abs(H_p2), label="2 stages (0.3, 0.5 Hz)")
#plt.loglog(f, np.abs(H_gas_3), label="3 stages (0.2, 0.3, 0.4 Hz)")
plt.loglog(f, np.abs(H_ip_after), label="IP (after counter-mass tuning)")
plt.xlabel("Frequency [Hz]", fontsize=13, fontname='Arial')
plt.ylabel(r"$|X/X_0|$", fontsize=13, fontname='Arial')
plt.title("Full Passive Isolation Chain (Horizontal)", fontsize=14, fontname='Arial')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()

plt.show()
