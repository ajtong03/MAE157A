import numpy as np

def product(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q1q2_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q1q2_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q1q2_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q1q2_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
     
    # Create a 4 element array containing the final quaternion
    prod_quaternion = np.array([q1q2_w, q1q2_x, q1q2_y, q1q2_z])
        
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return prod_quaternion
    
# function to compute the inverse of a quaternion
def inverse(q):
    mag = q[0]
    i = -q[1]
    j = -q[2]
    k = -q[3]
    q_inv = np.array([mag, i, j, k])

    return q_inv
    
# function to compute the error between desired quaternion and current quaternion
def error(qa, qd):
    qd_inv = inverse(qd)
    qe = product(qd_inv, qa)
 
    return qe

def deriv(q1, w):
    w_quat = [0, w[0], w[1], w[2]]
    q_deriv = 0.5 * product(q1, w_quat)
    return q_deriv

def euler_to_quat(phi, theta, psi):
    w = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    x = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    y = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    z = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    return np.array([w, x, y, z])