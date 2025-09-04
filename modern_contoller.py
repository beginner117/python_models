import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt

def demo_statespace_to_tf():
    # A lightly coupled 2-DoF toy system (positions & velocities)
    # x = [x1, v1, x2, v2]^T
    w1, z1 = 2*np.pi*0.8, 0.02
    w2, z2 = 2*np.pi*1.6, 0.02
    k12 = (0.1*w1*w2)  # weak coupling

    A = np.array([[0, 1, 0, 0],
                  [-w1**2, -2*z1*w1, k12, 0],
                  [0, 0, 0, 1],
                  [k12, 0, -w2**2, -2*z2*w2]])
    B = np.array([[0],[1],[0],[0]])   # actuate DoF-1 velocity channel
    C = np.array([[1,0,0,0],          # measure x1
                  [0,0,1,0]])         # measure x2
    D = np.zeros((2,1))

    # Frequency response H(jw) = C(jwI-A)^{-1}B + D
    w = np.logspace(-2, 2, 1000) * 2*np.pi
    H11, H21 = [], []
    for wi in w:
        G = C @ la.inv(1j*wi*np.eye(4) - A) @ B + D
        H11.append(G[0,0]); H21.append(G[1,0])
    H11 = np.array(H11); H21 = np.array(H21)

    f = w/(2*np.pi)
    plt.figure(figsize=(10,5))
    plt.semilogx(f, 20*np.log10(np.abs(H11)), label='|x1/u|')
    plt.semilogx(f, 20*np.log10(np.abs(H21)), label='|x2/u| (cross)')
    plt.title('State-space → TF: direct frequency response from (A,B,C,D)')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]'); plt.grid(True, which='both'); plt.legend()

    # ----------- Closed-loop with unity feedback -----------
    # Suppose we close the loop on x1 (first output) with unity gain:
    # u = -K*y1 + r  (here r=reference, K=1)
    # Form augmented system: Acl = A - B*K*C1
    K = np.array([[150.0]])   # feedback gain
    C1 = C[0:1,:]           # use only x1 as measurement
    Acl = A - B @ K @ C1
    Bcl = B.copy()          # reference input goes through same B
    Ccl = C.copy()
    Ccl = C - D @ K @ C1  # adjust C for feedback
    Dcl = D.copy()

    # Compute closed-loop frequency response
    H11_cl, H21_cl = [], []
    for wi in w:
        Gcl = Ccl @ la.inv(1j*wi*np.eye(4) - Acl) @ Bcl + Dcl
        H11_cl.append(Gcl[0,0]); H21_cl.append(Gcl[1,0])
    H11_cl = np.array(H11_cl); H21_cl = np.array(H21_cl)

    plt.figure(figsize=(10,5))
    plt.semilogx(f, 20*np.log10(np.abs(H11_cl)), label='|x1/r| (closed-loop)')
    plt.semilogx(f, 20*np.log10(np.abs(H21_cl)), label='|x2/r| (cross, closed-loop)')
    #plt.title('State-space → TF: closed-loop with unity feedback on x1')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.grid(True, which='both'); plt.legend()

        # ----------- Closed-loop : P vs PD feedback -----------
    # P: u = r - Kp*x1  (stiffens, little/no peak attenuation)
    Kp = 50.0
    Cx = C[0:1,:]                 # x1 measurement
    Acl_P = A - B @ np.array([[Kp]]) @ Cx

    # PD: u = r - Kp*x1 - Kd*v1  (adds damping, attenuates peak)
    Kd = 10.0
    Cv = np.array([[0,1,0,0]])    # v1 measurement
    Acl_PD = A - B @ (Kp*Cx + Kd*Cv)

    # Frequency responses from r -> outputs
    def frf(A_mat):
        H11_cl, H21_cl = [], []
        for wi in w:
            Gcl = C @ la.inv(1j*wi*np.eye(4) - A_mat) @ B + D
            H11_cl.append(Gcl[0,0]); H21_cl.append(Gcl[1,0])
        return np.array(H11_cl), np.array(H21_cl)

    H11_P,  H21_P  = frf(Acl_P)
    H11_PD, H21_PD = frf(Acl_PD)

    plt.figure(figsize=(10,5))
    plt.semilogx(f, 20*np.log10(np.abs(H11)),    linestyle='--', label='|x1/u| open')
    #plt.semilogx(f, 20*np.log10(np.abs(H11_P)),                 label='|x1/r| closed (P)')
    plt.semilogx(f, 20*np.log10(np.abs(H11_PD)),                label='|x1/r| closed (PD)')
    plt.title('Closed-loop on x1: P stiffens (shifts), PD damps (attenuates)')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.grid(True, which='both'); plt.legend()
 
