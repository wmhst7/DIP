import numpy as np


def get_epipole(F):
    v0, vec0 = np.linalg.eig(F)
    v1, vec1 = np.linalg.eig(np.transpose(F))
    e0 = vec0[:, np.argmin(v0)]
    e1 = vec1[:, np.argmin(v1)]
    return e0, e1


def rotate_mat(d, theta):
    co, si = np.cos(theta), np.sin(theta)
    t = 1 - co
    x, y = d[0], d[1]
    mat = np.array([t*x*x + co, t*x*y, si*y,
                    t*x*y, t*y*y + co, -si*x,
                    -si*y, si*x, co]).reshape(3, 3)
    return mat


def get_Rphi(p):
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def get_prematrix(F):
    e0, e1 = get_epipole(F)
    print('e0, e1:', e0, e1)
    d0 = np.array([-e0[1], e0[0], 0])
    Fd0 = np.dot(F, d0)
    d1 = np.array([-Fd0[1], Fd0[0], 0])
    theta0 = np.arctan(e0[2] / (d0[1] * e0[0] - d0[0] * e0[1]))
    theta1 = np.arctan(e1[2] / (d1[1] * e1[0] - d1[0] * e1[1]))
    # print('theta0:', theta0)
    Rd0th0 = rotate_mat(d0, theta0)
    Rd1th1 = rotate_mat(d1, theta1)
    # print('Rdoth0:', Rd0th0, 'e0', e0)
    e0n = Rd0th0.dot(e0)
    e1n = Rd1th1.dot(e1)
    phi0 = -np.arctan(e0n[1] / e0n[0])
    phi1 = -np.arctan(e1n[1] / e1n[0])
    Rphi0, Rphi1 = get_Rphi(phi0), get_Rphi(phi1)
    H0 = Rphi0.dot(Rd0th0)
    H1 = Rphi1.dot(Rd1th1)
    # m0 = np.mean(H0)
    # m1 = np.mean(H1)
    # H0 = H0 - m0
    # H1 = H1 - m1
    print('H0:', H0)
    print('H1:', H1)
    return H0, H1

