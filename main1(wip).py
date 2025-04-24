import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from math import sin, cos
from dynamics import dynamics
import quaternionfunc

# Try to import custom controller
try:
    import controller
    has_custom = True
except ImportError:
    has_custom = False

# --- conversions ------------------------------------------------

def euler_to_quat(phi, theta, psi):
    w = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
    x = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
    y = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
    z = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)
    return np.array([w, x, y, z])

def quat_to_euler(q):
    w, x, y, z = q
    phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    psi = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return phi, theta, psi

def attitude_from_acc(a_des, yaw):
    b3 = a_des / np.linalg.norm(a_des)
    c, s = cos(yaw), sin(yaw)
    b1d = np.array([c, s, 0.0])
    b2 = np.cross(b3, b1d); b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    R = np.column_stack((b1, b2, b3))
    t = np.trace(R)
    if t > 0:
        S = 2 * np.sqrt(t + 1.0)
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            S = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / S; qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S; qz = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S; qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S; qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

# --- trajectory --------------------------------------------------

def solve_poly(t_f, p0, v0, a0, pf, vf, af):
    A = np.array([
        [1, 0,      0,        0,         0,          0],
        [0, 1,      0,        0,         0,          0],
        [0, 0,      2,        0,         0,          0],
        [1, t_f,    t_f**2,   t_f**3,    t_f**4,     t_f**5],
        [0, 1,   2*t_f,   3*t_f**2,  4*t_f**3,   5*t_f**4],
        [0, 0,      2,    6*t_f,   12*t_f**2,  20*t_f**3]
    ])
    b = np.array([p0, v0, a0, pf, vf, af])
    return np.linalg.solve(A, b)

tf = 15.0
c_x = solve_poly(tf, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0)
def x_t(t): return sum(c_x[i] * t**i for i in range(6))
def vx_t(t): return sum(i * c_x[i] * t**(i-1) for i in range(1,6))
def ax_t(t): return sum(i*(i-1)*c_x[i] * t**(i-2) for i in range(2,6))

y_const, z_const, yaw_const = 0.0, 2.0, 0.0

dt = 1.0 / 50.0
dyn = dynamics(params=[9.81], dt=dt)
state = np.zeros(13)
state[0:3]  = [x_t(0), y_const, z_const]
state[3]    = vx_t(0)
state[6:10] = euler_to_quat(0, 0, yaw_const)

if has_custom:
    q_hist = [state[6:10].copy()] * 10
    q_des  = euler_to_quat(0, 0, yaw_const)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_zlim(0, 3)
trail, = ax.plot([], [], [], 'b-', lw=2)
arm1,  = ax.plot([], [], [], 'k-', lw=4)
arm2,  = ax.plot([], [], [], 'k-', lw=4)
history = []

gate = np.array([[-.5,0,-.25],[.5,0,-.25],[.5,0,.25],[-.5,0,.25],[-.5,0,-.25]])
Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])
gc = gate @ Rz.T + np.array([0, 0, z_const])
ax.plot(gc[:,0], gc[:,1], gc[:,2], 'g-', lw=2)

def update(i):
    global state, history, q_hist
    t = i * dt

    pd = np.array([x_t(min(t,tf)), y_const, z_const])
    vd = np.array([vx_t(t), 0.0, 0.0])
    ad = np.array([ax_t(t), 0.0, 0.0])
    e_p = pd - state[0:3]
    e_v = vd - state[3:6]
    a_des = ad + 1.5*e_p + 0.8*e_v + np.array([0, 0, dyn.g])
    T     = dyn.m * np.linalg.norm(a_des)

    q_d = attitude_from_acc(a_des, yaw_const)
    q_a = state[6:10]
    w   = state[10:13]

    qe = quaternionfunc.error(q_a, q_d)
    qe /= np.linalg.norm(qe)
    M0 = -5 * qe[1:] - 0.1 * w

    A_mat = np.array([[1,1,1,1], [0,dyn.l,0,-dyn.l], [-dyn.l,0,dyn.l,0], [dyn.c,-dyn.c,dyn.c,-dyn.c]])
    #f0 = np.linalg.solve(A_mat, np.hstack((T, M0)))
    f = np.linalg.solve(A_mat, np.hstack((T, M0)))
  
    
    if has_custom:
        # pass the correct initial thrust f0
        Kp_opt, Kd_opt = controller.naiveComputeGains(q_a, q_d, state, w, np.zeros(3), T)

       # M = controller.computeTorqueNaive(Kp_opt, Kd_opt, q_a, q_d, w, np.zeros(3))
        M = controller.computeTorqueNaive(-Kp_opt, Kd_opt, q_a, q_d, w, np.zeros(3))
    else:
        M = M0

    f = np.linalg.solve(A_mat, np.hstack((T, M)))
    state = dyn.propagate(state, f)

    if has_custom:
        q_hist.append(state[6:10].copy())
        if len(q_hist) > 10:
            q_hist.pop(0)

    history.append(state[0:3])
    pts = np.array(history)
    trail.set_data(pts[:,0], pts[:,1])
    trail.set_3d_properties(pts[:,2])

    phi, th, ps = quat_to_euler(state[6:10])
    offs = [(dyn.l,0,0), (-dyn.l,0,0), (0,dyn.l,0), (0,-dyn.l,0)]
    world = []
    for dx, dy, dz in offs:
        Xw = cos(ps)*cos(th)*dx + (cos(ps)*sin(phi)*sin(th)-cos(phi)*sin(ps))*dy \
             + (cos(phi)*cos(ps)*sin(th)+sin(phi)*sin(ps))*dz + state[0]
        Yw = sin(ps)*cos(th)*dx + (sin(ps)*sin(phi)*sin(th)+cos(phi)*cos(ps))*dy \
             + (cos(phi)*sin(ps)*sin(th)-cos(ps)*sin(phi))*dz + state[1]
        Zw = -sin(th)*dx + cos(th)*sin(phi)*dy + cos(phi)*cos(th)*dz + state[2]
        world.append((Xw, Yw, Zw))
    a, b, c, d = world
    arm1.set_data([a[0], c[0]], [a[1], c[1]])
    arm1.set_3d_properties([a[2], c[2]])
    arm2.set_data([b[0], d[0]], [b[1], d[1]])
    arm2.set_3d_properties([b[2], d[2]])

    return trail, arm1, arm2

frames = int(tf/dt) + 1
ani = animation.FuncAnimation(plt.gcf(), update, frames=frames, interval=dt*1000)
plt.show()