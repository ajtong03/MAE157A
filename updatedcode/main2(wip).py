import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from math import sin, cos, tan
from dynamics import quadrotor

# Polynomial trajectory setup for x position

def solve_polynomial_coefficients(t_f, p0, v0, a0, pf, vf, af):
    A = np.array([
        [1,    0,    0,      0,       0,        0],
        [0,    1,    0,      0,       0,        0],
        [0,    0,    2,      0,       0,        0],
        [1,   t_f,  t_f**2, t_f**3,  t_f**4,   t_f**5],
        [0,    1,  2*t_f,  3*t_f**2,4*t_f**3, 5*t_f**4],
        [0,    0,    2,   6*t_f,  12*t_f**2,20*t_f**3]
    ])
    b = np.array([p0, v0, a0, pf, vf, af])
    return np.linalg.solve(A, b)

# Define final time and boundary conditions
tf = 15.0  # seconds
# x: from 1.5 m to 0 m, velocity 0->3 m/s, accel 0
c_x = solve_polynomial_coefficients(tf, 1.5, 0.0, 0.0, 0.0, 3.0, 0.0)

def x_t(t):
    return sum(c_x[i] * t**i for i in range(6))

def vx_t(t):
    return sum(i * c_x[i] * t**(i-1) for i in range(1,6))

def ax_t(t):
    return sum(i*(i-1) * c_x[i] * t**(i-2) for i in range(2,6))

# For this demo, hold y and z constant or simple linear
y_const = 0.0
z_const = 2.0
yaw_const = 0.0

# Setup quad
dt = 1.0/50.0
qd = quadrotor(Ts=dt, USE_PID=1)
# Initialize state to match polynomial start conditions
state = [0.0]*12
# Set initial velocities and positions from trajectory
state[6]  = vx_t(0.0)        # x_dot
state[7]  = 0.0               # y_dot
state[8]  = 0.0               # z_dot
state[9]  = x_t(0.0)         # x position
state[10] = y_const          # y position
state[11] = z_const          # z position

# Setup figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1,2)
ax.set_ylim(-1,2)
ax.set_zlim(0,3)
# Draw stationary gate
from math import cos as _cos, sin as _sin
gate_center = np.array([0.0, 0.0, z_const])
gate_yaw = np.pi/2
w, h = 2.0, 0.5
corners = np.array([[-w/2,0,-h/2],[w/2,0,-h/2],[w/2,0,h/2],[-w/2,0,h/2],[-w/2,0,-h/2]])
Rz = np.array([[_cos(gate_yaw), -_sin(gate_yaw),0],[_sin(gate_yaw),_cos(gate_yaw),0],[0,0,1]])
cw = (Rz @ corners.T).T + gate_center
ax.plot(cw[:,0],cw[:,1],cw[:,2],'g-',lw=2)

line, = ax.plot([],[],[], 'b-', lw=2)
arm1, = ax.plot([],[],[], 'k-', lw=4)
arm2, = ax.plot([],[],[], 'k-', lw=4)
path = []
arm_len = qd.l

# Animation update
def update(frame):
    global state, path
    t = frame * dt
    # compute desired trajectory
    x_d = x_t(min(t, tf))
    y_d = y_const
    z_d = z_const
    yaw_d = yaw_const
    # setpoints
    qd.des_xyz(x_d, y_d, z_d)
    qd.psi_des = yaw_d
    # control
    qd.PID_position()
    qd.PID_attitude()
    qd.PID_rate()
    # propagate
    state, _ = qd.step(state, qd.input_vector)
    print(f"t={t:.2f}, pos=({state[9]:.2f},{state[10]:.2f},{state[11]:.2f})")
    path.append(state[9:12])
    pts = np.array(path)
    line.set_data(pts[:,0], pts[:,1]); line.set_3d_properties(pts[:,2])
    # draw arms
    phi,theta,psi = state[:3]
    offsets = [(arm_len,0,0),(-arm_len,0,0),(0,arm_len,0),(0,-arm_len,0)]
    world = []
    for dx,dy,dz in offsets:
        Xw = cos(psi)*cos(theta)*dx + (cos(psi)*sin(phi)*sin(theta)-cos(phi)*sin(psi))*dy + \
             (cos(phi)*cos(psi)*sin(theta)+sin(phi)*sin(psi))*dz + state[9]
        Yw = sin(psi)*cos(theta)*dx + (sin(psi)*sin(phi)*sin(theta)+cos(phi)*cos(psi))*dy + \
             (cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi))*dz + state[10]
        Zw = -sin(theta)*dx + cos(theta)*sin(phi)*dy + cos(phi)*cos(theta)*dz + state[11]
        world.append((Xw,Yw,Zw))
    a,b,c,d = world
    arm1.set_data([a[0],c[0]],[a[1],c[1]]); arm1.set_3d_properties([a[2],c[2]])
    arm2.set_data([b[0],d[0]],[b[1],d[1]]); arm2.set_3d_properties([b[2],d[2]])
    return line, arm1, arm2

# Create animation
frames = int(tf/dt) + 1
ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=False)
plt.show()
