import numpy as numpy
import matplotlib.pyplot as plt

def solve_polynomial_coefficients(t_f, p0, v0, a0, pf, vf, af):
#5th order polynomial coefficients , maybe 7th order instead?

    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5],
        [0, 1, 2 * t_f, 3 * t_f**2, 4 * t_f**3, 5 * t_f**4],
        [0, 0, 2, 6 * t_f, 12 * t_f**2, 20 * t_f**3]
        # [0, 0, 0, 6, 24*t_f, 60*t_f**2]
        # [0, 0, 0, 0 , 24,120*t_f]
    ])
    b = np.array([p0, v0, a0, pf, vf, af])
    c = np.linalg.solve(A, b)
    return c
# Approach Segment 
# Goal is to have drone go through origin (gate location)
#Recommended to have 0 acceleration through the gate
# mass_drone = 0.399 kg
# X-DIRECTION 
tf = 15  # seconds
c_x = solve_polynomial_coefficients(tf, 1.5, 0, 0, 0, 3, 0) #BCs
print("x-Coeffs:", c_x)

def x_t(t):
    return c_x[0] + c_x[1] * t + c_x[2] * t**2 + c_x[3] * t**3 + c_x[4] * t**4 + c_x[5] * t**5

def vx_t(t):
    return c_x[1] + 2 * c_x[2] * t + 3 * c_x[3] * t**2 + 4 * c_x[4] * t**3 + 5 * c_x[5] * t**4

def ax_t(t):
    return 2 * c_x[2] + 6 * c_x[3] * t + 12 * c_x[4] * t**2 + 20 * c_x[5] * t**3
