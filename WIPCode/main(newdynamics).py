import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from math import sin, cos
from dynamics import dynamics
import quaternionfunc

# --- utility conversions -----------------------------------------

def euler_to_quat(phi, theta, psi):
    w = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
    x = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
    y = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
    z = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)
    return np.array([w, x, y, z])

def quat_to_euler(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    # roll (φ)
    phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    # pitch (θ)
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    # yaw (ψ)
    psi = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return phi, theta, psi

def solve_polynomial_coefficients(t_f, p0, v0, a0, pf, vf, af):
    A = np.array([
        [1,    0,      0,        0,         0,          0],
        [0,    1,      0,        0,         0,          0],
        [0,    0,      2,        0,         0,          0],
        [1,   t_f,   t_f**2,   t_f**3,    t_f**4,     t_f**5],
        [0,    1,   2*t_f,   3*t_f**2,  4*t_f**3,   5*t_f**4],
        [0,    0,      2,    6*t_f,   12*t_f**2,  20*t_f**3]
    ])
    b = np.array([p0, v0, a0, pf, vf, af])
    return np.linalg.solve(A, b)

# desired trajectory in X
tf = 15.0
c_x = solve_polynomial_coefficients(tf, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0)
def x_t(t):   return sum(c_x[i] * t**i for i in range(6))
def vx_t(t):  return sum(i * c_x[i] * t**(i-1) for i in range(1,6))
def ax_t(t):  return sum(i*(i-1)*c_x[i] * t**(i-2) for i in range(2,6))

y_const, z_const, yaw_const = 0.0, 2.0, 0.0

# build desired attitude quaternion from accel + yaw
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

# --- initialize dynamics & state --------------------------------

dt = 1.0 / 50.0
dyn = dynamics(params=[9.81], dt=dt)

# state = [ x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz ]
state = np.zeros(13)
state[0:3]   = [x_t(0.0), y_const, z_const]
state[3]     = vx_t(0.0)
state[6:10]  = euler_to_quat(0.0, 0.0, yaw_const)

# --- set up 3D gate + drone plot -------------------------------

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 2); ax.set_ylim(-1, 2); ax.set_zlim(0, 3)

# draw gate
gate_center = np.array([0.0, 0.0, z_const])
gate_yaw    = np.pi/2
w, h = 1.0, 0.5
corners = np.array([[-w/2,0,-h/2],[w/2,0,-h/2],[w/2,0,h/2],
                    [-w/2,0,h/2],[-w/2,0,-h/2]])
Rz = np.array([[ cos(gate_yaw), -sin(gate_yaw), 0],
               [ sin(gate_yaw),  cos(gate_yaw), 0],
               [0,0,1]])
cw = (Rz @ corners.T).T + gate_center
ax.plot(cw[:,0], cw[:,1], cw[:,2], 'g-', lw=2)

# drone trail + arms
trail, = ax.plot([], [], [], 'b-', lw=2)
arm1,  = ax.plot([], [], [], 'k-', lw=4)
arm2,  = ax.plot([], [], [], 'k-', lw=4)
history = []

def update(frame):
    global state, history
    t = frame * dt

    # --- desired traj + PD position control ---------------------
    pd = np.array([x_t(min(t,tf)), y_const, z_const])
    vd = np.array([vx_t(t),       0.0,     0.0])
    ad = np.array([ax_t(t),       0.0,     0.0])
    e_pos = pd - state[0:3]
    e_vel = vd - state[3:6]
    # gains: 1.5 / 0.8 chosen to match old PD
    a_des = ad + 1.5*e_pos + 0.8*e_vel + np.array([0,0,dyn.g])
    T     = dyn.m * np.linalg.norm(a_des)

    # --- attitude setpoint (quaternion) -------------------------
    q_d = attitude_from_acc(a_des, yaw_const)
    q_a = state[6:10]
    w   = state[10:13]

    # simple quaternion-PD torque
    q_err = quaternionfunc.error(q_a, q_d)
    q_err /= np.linalg.norm(q_err)
    qe_vec = q_err[1:]
    Kp_att = np.diag([5.0, 5.0, 5.0])
    Kd_att = np.diag([0.1, 0.1, 0.1])
    M = -Kp_att @ qe_vec - Kd_att @ w

    # --- solve for per-rotor thrusts f0..f3 ----------------------
    l = dyn.l; c = dyn.c
    A = np.array([
        [   1,    1,    1,    1],
        [   0,    l,    0,   -l],
        [  -l,    0,    l,    0],
        [   c,   -c,    c,   -c]
    ])
    b = np.hstack((T, M))
    f = np.linalg.solve(A, b)

    # --- propagate dynamics -------------------------------------
    state = dyn.propagate(state, f)

    # --- record + draw trail ------------------------------------
    history.append(state[0:3])
    pts = np.array(history)
    trail.set_data(pts[:,0], pts[:,1])
    trail.set_3d_properties(pts[:,2])

    # --- compute rotor-arm endpoints ----------------------------
    phi, th, ps = quat_to_euler(state[6:10])
    L = dyn.l
    offs = [( L,0,0),(-L,0,0),(0, L,0),(0,-L,0)]
    world = []
    for dx,dy,dz in offs:
        Xw = cos(ps)*cos(th)*dx + (cos(ps)*sin(phi)*sin(th)-cos(phi)*sin(ps))*dy \
             + (cos(phi)*cos(ps)*sin(th)+sin(phi)*sin(ps))*dz + state[0]
        Yw = sin(ps)*cos(th)*dx + (sin(ps)*sin(phi)*sin(th)+cos(phi)*cos(ps))*dy \
             + (cos(phi)*sin(ps)*sin(th)-cos(ps)*sin(phi))*dz + state[1]
        Zw = -sin(th)*dx + cos(th)*sin(phi)*dy + cos(phi)*cos(th)*dz + state[2]
        world.append((Xw,Yw,Zw))

    a,b,c,d = world
    arm1.set_data([a[0], c[0]], [a[1], c[1]])
    arm1.set_3d_properties([a[2], c[2]])
    arm2.set_data([b[0], d[0]], [b[1], d[1]])
    arm2.set_3d_properties([b[2], d[2]])

    return trail, arm1, arm2

# run
frames = int(tf/dt) + 1
ani    = animation.FuncAnimation(fig, update, frames=frames, interval=dt*1000)
plt.show()
