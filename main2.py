import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.animation as animation  
from mpl_toolkits.mplot3d import Axes3D  
from math import sin, cos  
import trajectory 
from dynamics import dynamics  
import quaternionfunc  

try:
    import controller
    has_custom = True
except ImportError:
    has_custom = False

# --- conversion utilities ------------------------------------------------
def euler_to_quat(phi, theta, psi):
    """
    Convert Euler angles (roll=phi, pitch=theta, yaw=psi) to a quaternion [w, x, y, z].
    """
    w = cos(phi/2)*cos(theta/2)*cos(psi/2) + sin(phi/2)*sin(theta/2)*sin(psi/2)
    x = sin(phi/2)*cos(theta/2)*cos(psi/2) - cos(phi/2)*sin(theta/2)*sin(psi/2)
    y = cos(phi/2)*sin(theta/2)*cos(psi/2) + sin(phi/2)*cos(theta/2)*sin(psi/2)
    z = cos(phi/2)*cos(theta/2)*sin(psi/2) - sin(phi/2)*sin(theta/2)*cos(psi/2)
    return np.array([w, x, y, z])

def quat_to_euler(q):
    """
    Convert quaternion q = [w, x, y, z] back to Euler angles (phi, theta, psi).
    """
    w, x, y, z = q
    phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    psi = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return phi, theta, psi

def attitude_from_acc(a_des, yaw):
    """
    Build a desired attitude quaternion from a desired acceleration vector and yaw.
    """
    b3 = a_des / np.linalg.norm(a_des)
    c, s = cos(yaw), sin(yaw)
    b1d = np.array([c, s, 0.0])
    b2 = np.cross(b3, b1d); b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    R = np.column_stack((b1, b2, b3))
    t = np.trace(R)
    if t > 0:
        S = 2 * np.sqrt(t + 1.0)
        qw = 0.25 * S; qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S; qz = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            S = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / S; qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S; qz = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / S; qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S; qz = (R[1,2] + R[2,1]) / S
        else:
            S = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / S; qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S; qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

# --- user-configurable initial conditions ---------------------------------
initial_t = 0.0            # start time along trajectory
t0_roll = 0.0             # initial roll (rad)
initial_tilt = 0.0         # initial pitch (rad)
initial_yaw = 0.0          # initial heading (rad)

# --- simulation setup ------------------------------------------------
dt = 1.0/50.0
dyn = dynamics(params=[9.81], dt=dt)
state = np.zeros(13)
# initialize state from trajectory and orientation from user inputs
state[0:3]  = [trajectory.x_t(initial_t), trajectory.y_t(initial_t), trajectory.z_t(initial_t)]
state[3:6]  = [trajectory.vx_t(initial_t), trajectory.vy_t(initial_t), trajectory.vz_t(initial_t)]
state[6:10] = euler_to_quat(t0_roll, initial_tilt, initial_yaw)
state[10:13] = np.zeros(3)

# --- plotting setup ------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(0, 4)
trail, = ax.plot([], [], [], 'b-', lw=2)
arm1, = ax.plot([], [], [], 'k-', lw=2)
arm2, = ax.plot([], [], [], 'k-', lw=2)
history = []
axis_arrows = []
# gate at z=1 rotated 90° about Y
gate = np.array([[-.5,0,-.25],[.5,0,-.25],[.5,0,.25],[-.5,0,.25],[-.5,0,-.25]])
ty = np.array([[0,0,1],[0,1,0],[-1,0,0]])
gate_pts = gate @ ty.T + np.array([0,0,1])
ax.plot(gate_pts[:,0], gate_pts[:,1], gate_pts[:,2], 'g-', lw=2)

# --- update function ------------------------------------------------
def update(i):
    global state, history, axis_arrows
    t = i * dt
    # clamp simulation time to available trajectory
    tf_total = trajectory.tf + trajectory.tf1
    t_clamped = min(t, tf_total)
    # pick desired segment of trajectory
    if t_clamped <= trajectory.tf:
        pd = np.array([trajectory.x_t(t_clamped), trajectory.y_t(t_clamped), trajectory.z_t(t_clamped)])
        vd = np.array([trajectory.vx_t(t_clamped), trajectory.vy_t(t_clamped), trajectory.vz_t(t_clamped)])
        ad = np.array([trajectory.ax_t(t_clamped), trajectory.ay_t(t_clamped), trajectory.az_t(t_clamped)])
    else:
        tau = t_clamped - trajectory.tf
        pd = np.array([trajectory.x_t1(tau), trajectory.y_t1(tau), trajectory.z_t1(tau)])
        vd = np.array([trajectory.vx_t1(tau), trajectory.vy_t1(tau), trajectory.vz_t1(tau)])
        ad = np.array([trajectory.ax_t1(tau), trajectory.ay_t1(tau), trajectory.az_t1(tau)])
    # compute control thrust & torque
    e_p = pd - state[0:3]
    e_v = vd - state[3:6]
    a_des = ad + 1.5*e_p + 0.8*e_v + np.array([0,0,dyn.g])
    T = dyn.m * np.linalg.norm(a_des)
    q_d = attitude_from_acc(a_des, initial_yaw)
    q_a = state[6:10]
    w   = state[10:13]
    qe  = quaternionfunc.error(q_a, q_d); qe /= np.linalg.norm(qe)
    M0  = -5 * qe[1:] - 0.1 * w
    # allow custom controller
    if has_custom:
        M = controller.computeTorqueNaive(-5*np.eye(3), 0.1*np.eye(3), q_a, q_d, w, np.zeros(3))
    else:
        M = M0
    # mix thrust and torque into rotor forces
    A = np.array([[1,1,1,1], [0,dyn.l,0,-dyn.l], [-dyn.l,0,dyn.l,0], [dyn.c,-dyn.c,dyn.c,-dyn.c]])
    f = np.linalg.solve(A, np.hstack((T, M)))
    # propagate full dynamics (updates pos, vel, quat, omega)
    state[:] = dyn.propagate(state, f)
    # update trail history
    history.append(state[0:3].copy())
    pts = np.array(history)
    trail.set_data(pts[:,0], pts[:,1]); trail.set_3d_properties(pts[:,2])
    # draw arms based on new orientation
    phi, th, ps = quat_to_euler(state[6:10])
    offs = [(dyn.l,0,0),(-dyn.l,0,0),(0,dyn.l,0),(0,-dyn.l,0)]
    world = []
    for dx,dy,dz in offs:
        Xw = cos(ps)*cos(th)*dx + (cos(ps)*sin(phi)*sin(th)-cos(phi)*sin(ps))*dy \
             + (cos(phi)*cos(ps)*sin(th)+sin(phi)*sin(ps))*dz + state[0]
        Yw = sin(ps)*cos(th)*dx + (sin(ps)*sin(phi)*sin(th)+cos(phi)*cos(ps))*dy \
             + (cos(phi)*sin(ps)*sin(th)-cos(ps)*sin(phi))*dz + state[1]
        Zw = -sin(th)*dx + cos(th)*sin(phi)*dy + cos(phi)*cos(th)*dz + state[2]
        world.append((Xw,Yw,Zw))
    a,b,c,d = world
    arm1.set_data([a[0],c[0]],[a[1],c[1]]); arm1.set_3d_properties([a[2],c[2]])
    arm2.set_data([b[0],d[0]],[b[1],d[1]]); arm2.set_3d_properties([b[2],d[2]])
    # update orientation arrows
    for art in axis_arrows:
        art.remove()
    axis_arrows = []
    Rmat = dyn.quat_to_rot(state[6:10])
    pos = state[0:3]
    axis_arrows.append(ax.quiver(pos[0],pos[1],pos[2],Rmat[0,0],Rmat[1,0],Rmat[2,0],length=0.3))
    axis_arrows.append(ax.quiver(pos[0],pos[1],pos[2],Rmat[0,1],Rmat[1,1],Rmat[2,1],length=0.3))
    axis_arrows.append(ax.quiver(pos[0],pos[1],pos[2],Rmat[0,2],Rmat[1,2],Rmat[2,2],length=0.3))
    return trail, arm1, arm2

# --- run animation ------------------------------------------------
frames = int((trajectory.tf + trajectory.tf1)/dt) + 1
ani = animation.FuncAnimation(plt.gcf(), update, frames=frames, interval=dt*1000)
plt.show()
