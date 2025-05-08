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
    w, x, y, z = q
    q_inv = np.array([w, -x, -y, -z])

    return q_inv
    
# function to compute the error between desired quaternion and current quaternion
def error(qa, qd):
    qd_inv = inverse(qd)
    qe = product(qd_inv, qa)
    # if the magnitude of qe is 0, the result should be the identity quaternion
    if np.linalg.norm(qe) == 0:
        return np.array([1, 0, 0, 0])
    qe = qe / np.linalg.norm(qe)

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

def quat_to_euler(q):
    w, x, y, z = q
    phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    psi = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return phi, theta, psi

def unitQuat(q):
    if abs(np.linalg.norm(q)) == 0:
      return np.array([0.0, 0.0, 0.0, 0.0])  # Avoid division by zero, return identity quaternion
  
    return q / np.linalg.norm(q)

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])

# https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
def R_to_quat(R):

    # R must be an array so can do transpose
    if type(R) != np.ndarray:
        R = np.array(R)

    # take transpose
    r0, r1, r2 = R.T
    m00, m01, m02 = r0
    m10, m11, m12 = r1
    m20, m21, m22 = r2
    
    if m22 < 0:
        if m00 > m11:
            t = 1 + m00 - m11 - m22
            q = np.array([t, m01 + m10, m20 + m02, m12 - m21])
        else:
            t = 1 - m00 + m11 - m22
            q = np.array([m01 + m10, t, m12 + m21, m20 - m02])
    else:
        if m00 < -m11:
            t = 1 - m00 - m11 + m22
            q = np.array([m20 + m02, m12 + m21, t, m01 - m10])
        else:
            t = 1 + m00 + m11 + m22
            q = np.array([m12 - m21, m20 - m02, m01 - m10, t])

    q *= 1 / 2 / np.sqrt(t)
    quat = np.array([q[3], q[0], q[1], q[2]])
    return quat / np.linalg.norm(quat)

