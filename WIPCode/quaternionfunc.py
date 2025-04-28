import numpy as np

def product(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Quaternion inverse (conjugate for unit quaternions)
def inverse(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

# Error quaternion: q_err = q_des^{-1} * q_act
def error(q_act, q_des):
    qd_inv = inverse(q_des)
    return product(qd_inv, q_act)