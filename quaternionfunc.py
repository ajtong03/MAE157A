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
def inverse(q1):
    mag = q1(0)
    vec = -q1(1)
    q_inv = np.array([mag, vec])

    return q_inv
    
# function to compute the error between desired quaternion and current quaternion
def error(q1, qd):
    qd_inv = inverse(qd)
    qe = product(qd_inv, q1)
 
    return qe

def deriv(q1, w):
    q_deriv = 0.5 * product(q1, (0, w))
    return q_deriv