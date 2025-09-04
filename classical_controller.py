import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt

def bode_mag_phase(num, den, w):
    w, h = sig.freqs(num, den, worN=w)
    mag_db = 20*np.log10(np.abs(h))
    ph_deg = np.unwrap(np.angle(h))*180/np.pi
    return mag_db, ph_deg

def tf_series(num1, den1, num2, den2):
    return np.polymul(num1, num2), np.polymul(den1, den2)

def tf_feedback(num_ol, den_ol):  # 1/(1+L) with unity feedback for closed-loop from ref->output: L/(1+L)
    # Closed-loop T = L/(1+L)
    num_cl = num_ol
    den_cl = np.polyadd(den_ol, num_ol)
    return num_cl, den_cl

def demo_pole_zero_nearcancel():
    w = np.logspace(-2, 2, 1000) * 2*np.pi  # rad/s

    # Plant: underdamped 2nd-order resonance (e.g., pendulum)
    w0 = 2*np.pi*0.1
    zeta = 0.02
    P_num = [w0**2]
    P_den = [1, 2*zeta*w0, w0**2]

    # Controller: lead-like zero near w0, with high-freq pole for roll-off
    K = 5.0
    z = w0         # controller zero near resonance
    p = 2*np.pi*10 # faster pole to limit HF gain
    C_num = K * np.array([1/z, 1.0])     # K*(s/z + 1)
    C_den = np.array([1/p, 1.0])         # (s/p + 1)

    # Open-loop: L = C*P
    L_num, L_den = tf_series(C_num, C_den, P_num, P_den)

    # Closed-loop complementary sensitivity T = L/(1+L)
    T_num, T_den = tf_feedback(L_num, L_den)

    # Bode plots
    magP, phP = bode_mag_phase(P_num, P_den, w)
    magL, phL = bode_mag_phase(L_num, L_den, w)
    magT, phT = bode_mag_phase(T_num, T_den, w)

    f = w/(2*np.pi)
    plt.figure(figsize=(10,5))
    plt.semilogx(f, magP, label='Plant |G|')
    plt.semilogx(f, magL, label='Open-loop ')
    plt.semilogx(f, magT, label='Closed-loop ')
    plt.axvline(w0/(2*np.pi), color='gray', ls='--')
    plt.title('pole attenuation (classical control theory)')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]'); plt.grid(True, which='both'); plt.legend()
    

    plt.figure(figsize=(10,5))
    plt.semilogx(f, phL, label='∠L')
    plt.semilogx(f, phT, label='∠T')
    plt.title('Phase around resonance with lead compensation')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Phase [deg]'); plt.grid(True, which='both'); plt.legend()
