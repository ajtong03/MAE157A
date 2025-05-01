import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sin,cos, tan
from dynamics import dynamics   # import the class, not the module
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

#-------polynomial tracking --------------------------------
def solve_polynomial_coefficients(t_f, p0, v0, a0, j0, pf, vf, af, jf):
# 7th order polynomial
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], # c_0 = p0
        [0, 1, 0, 0, 0, 0, 0, 0], # c_1 = v0
        [0, 0, 2, 0, 0, 0, 0, 0], # c_2 = a0/2
        [0, 0, 0, 6, 0, 0, 0, 0], # c_3 = j0/6
        [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5, t_f**6, t_f**7],
        [0, 1, 2 * t_f, 3 * t_f**2, 4 * t_f**3, 5 * t_f**4, 6 * t_f**5, 7 * t_f**6],
        [0, 0, 2, 6 * t_f, 12 * t_f**2, 20 * t_f**3, 30 * t_f**4, 42 * t_f**5],
        [0, 0, 0, 6, 24 * t_f, 60 * t_f**2, 120 * t_f**3, 210 * t_f**4]
        
    ])
    b = np.array([p0, v0, a0, j0, pf, vf, af, jf])
    c = np.linalg.solve(A, b)
    return c

# Approach Segment 
# Goal is to have drone go through origin (gate location)
# Recommended to have 0 acceleration through the gate and thrust perpendicular to gate's side
# x - axis
tf = 8 # seconds
time_approach = np.linspace(0, tf, 200)

c_x = solve_polynomial_coefficients(tf, 1.5, 0, 0, 0, 0, 1.25, 0.5, 0 )
# acceleration must be zero (so it doesnt move forward or backwards going through gate)
print("x-coeffs:", c_x)
def x_t(t):
    return c_x[0] + c_x[1] * t + c_x[2] * t**2 + c_x[3] * t**3 + c_x[4] * t**4 + c_x[5] * t**5 + c_x[6] * t**6 + c_x[7] * t**7
def vx_t(t):
    return c_x[1] + 2 * c_x[2] * t + 3 * c_x[3] * t**2 + 4 * c_x[4] * t**3 + 5 * c_x[5] * t**4 + 6 * c_x[6] * t**5 + 7 * c_x[7] * t**6

def ax_t(t):
    return 2 * c_x[2] + 6 * c_x[3] * t + 12 * c_x[4] * t**2 + 20 * c_x[5] * t**3 + 30 * c_x[6] * t**4 + 42 * c_x[7] * t**5 

def jx_t(t):
    return 6 * c_x[3] + 24 * c_x[4] * t + 60 * c_x[5] * t**2 + 120 * c_x[6] * t**3 + 210 * c_x[7] * t**4


# y - dir
# focus on y accel (make sure orientation lines up)
c_y = solve_polynomial_coefficients(tf, -1.5 , 0 ,0, 0, 0, 1.4142,0.5, 0 )
print("y-coeffs:", c_y)
def y_t(t):
    return c_y[0] + c_y[1] * t + c_y[2] * t**2 + c_y[3] * t**3 + c_y[4] * t**4 + c_y[5] * t**5 + c_y[6] * t**6 + c_y[7] * t**7
def vy_t(t):
    return c_y[1] + 2 * c_y[2] * t + 3 * c_y[3] * t**2 + 4 * c_y[4] * t**3 + 5 * c_y[5] * t**4 + 6 * c_y[6] * t**5 + 7 * c_y[7] * t**6

def ay_t(t):
    return 2 * c_y[2] + 6 * c_y[3] * t + 12 * c_y[4] * t**2 + 20 * c_y[ 5] * t**3 + 30 * c_y[6] * t**4 + 42 * c_y[7] * t**5 

def jy_t(t):
    return 6 * c_y[3] + 24 * c_y[4] * t + 60 * c_y[5] * t**2 + 120 * c_y[6] * t**3 + 210 * c_y[7] * t**4


# z - dir 
# accel how fast v_z is accelerating up/down
c_z = solve_polynomial_coefficients(tf, 0, 0, 0, 0, 1, 0, 0, 0 )
print("z-coeffs:", c_z)
# remember acceleration <==> orientation/thrust (yaw in z-dir rotates plane left and right)
# needs to counteract gravity in this case
def z_t(t):
    return c_z[0] + c_z[1] * t + c_z[2] * t**2 + c_z[3] * t**3 + c_z[4] * t**4 + c_z[5] * t**5 + c_z[6] * t**6 + c_z[7] * t**7
def vz_t(t):
    return c_z[1] + 2 * c_z[2] * t + 3 * c_z[3] * t**2 + 4 * c_z[4] * t**3 + 5 * c_z[5] * t**4 + 6 * c_z[6] * t**5 + 7 * c_z[7] * t**6

def az_t(t):
    return 2 * c_z[2] + 6 * c_z[3] * t + 12 * c_z[4] * t**2 + 20 * c_z[ 5] * t**3 + 30 * c_z[6] * t**4 + 42 * c_z[7] * t**5 

def jz_t(t):
    return 6 * c_z[3] + 24 * c_z[4] * t + 60 * c_z[5] * t**2 + 120 * c_z[6] * t**3 + 210 * c_z[7] * t**4


# Departure Segment 
# Initial Boundary Conditions must match final BCs from approach segment
tf1 =  5 # seconds
time_departure = np.linspace(0, tf1, 200)

# x - axis
c_x1 = solve_polynomial_coefficients(tf1, 0, 1.25, 0.5, 0, -1.5, 0, 0, 0 )
#
print("x-coeffs:", c_x1)
def x_t1(t):
    return c_x1[0] + c_x1[1] * t + c_x1[2] * t**2 + c_x1[3] * t**3 + c_x1[4] * t**4 + c_x1[5] * t**5 + c_x1[6] * t**6 + c_x1[7] * t**7
def vx_t1(t):
    return c_x1[1] + 2 * c_x1[2] * t + 3 * c_x1[3] * t**2 + 4 * c_x1[4] * t**3 + 5 * c_x1[5] * t**4 + 6 * c_x1[6] * t**5 + 7 * c_x1[7] * t**6

def ax_t1(t):
    return 2 * c_x1[2] + 6 * c_x1[3] * t + 12 * c_x1[4] * t**2 + 20 * c_x1[5] * t**3 + 30 * c_x1[6] * t**4 + 42 * c_x1[7] * t**5 

def jx_t1(t):
    return 6 * c_x1[3] + 24 * c_x1[4] * t + 60 * c_x1[5] * t**2 + 120 * c_x1[6] * t**3 + 210 * c_x1[7] * t**4


# y - dir
c_y1 = solve_polynomial_coefficients(tf1, 0 , 1.4142 ,0.5, 0, 1.5, 0, 0, 0 )
print("y-coeffs:", c_y1)
def y_t1(t):
    return c_y1[0] + c_y1[1] * t + c_y1[2] * t**2 + c_y1[3] * t**3 + c_y1[4] * t**4 + c_y1[5] * t**5 + c_y1[6] * t**6 + c_y1[7] * t**7
def vy_t1(t):
    return c_y1[1] + 2 * c_y1[2] * t + 3 * c_y1[3] * t**2 + 4 * c_y1[4] * t**3 + 5 * c_y1[5] * t**4 + 6 * c_y1[6] * t**5 + 7 * c_y1[7] * t**6

def ay_t1(t):
    return 2 * c_y1[2] + 6 * c_y1[3] * t + 12 * c_y1[4] * t**2 + 20 * c_y1[ 5] * t**3 + 30 * c_y1[6] * t**4 + 42 * c_y1[7] * t**5 

def jy_t1(t):
    return 6 * c_y1[3] + 24 * c_y1[4] * t + 60 * c_y1[5] * t**2 + 120 * c_y1[6] * t**3 + 210 * c_y1[7] * t**4


# z - dir 
c_z1 = solve_polynomial_coefficients(tf1, 1, 0 ,0, 0, 0, 0, 0, 0 )
print("z-coeffs:", c_z1)
# remember acceleration <==> orientation/thrust (yaw in z-dir rotates plane left and right)
# needs to counteract gravity in this case 
def z_t1(t):
    return c_z1[0] + c_z1[1] * t + c_z1[2] * t**2 + c_z1[3] * t**3 + c_z1[4] * t**4 + c_z1[5] * t**5 + c_z1[6] * t**6 + c_z1[7] * t**7
def vz_t1(t):
    return c_z1[1] + 2 * c_z1[2] * t + 3 * c_z1[3] * t**2 + 4 * c_z1[4] * t**3 + 5 * c_z1[5] * t**4 + 6 * c_z1[6] * t**5 + 7 * c_z1[7] * t**6

def az_t1(t):
    return 2 * c_z1[2] + 6 * c_z1[3] * t + 12 * c_z1[4] * t**2 + 20 * c_z1[ 5] * t**3 + 30 * c_z1[6] * t**4 + 42 * c_z1[7] * t**5 

def jz_t1(t):
    return 6 * c_z1[3] + 24 * c_z1[4] * t + 60 * c_z1[5] * t**2 + 120 * c_z1[6] * t**3 + 210 * c_z1[7] * t**4



# Separately plot polynomials (best to see distance it covers)
'''
# x - dir
plt.figure(figsize=(8, 6))
plt.plot(time_departure, [x_t(t) for t in time_departure], label="Position")
plt.plot(time_departure, [vx_t(t) for t in time_departure], label="Velocity")
#plt.plot(time_values, [ax_t(t) for t in time_approach], label="Acceleration")
#plt.plot(time_values, [jx_t(t) for t in time_values], label = "Jerk")
plt.title("X-Direction Trajectory (departure segment)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# y-dir
plt.figure(figsize=(8, 6))
plt.plot(time_departure, [y_t(t) for t in time_departure], label="Position")
plt.plot(time_departure, [vy_t(t) for t in time_departure], label="Velocity")
plt.plot(time_departure, [ay_t(t) for t in time_departure], label="Acceleration")
plt.title("Y-Direction Trajectory (approach segment)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# z-dir

plt.figure(figsize=(8, 6))
plt.plot(time_departure, [z_t(t) for t in time_departure], label="Position")
plt.plot(time_departure, [vz_t(t) for t in time_departure], label="Velocity")
plt.plot(time_departure, [az_t(t) for t in time_departure], label="Acceleration")
plt.title("Z-Direction Trajectory (departure segment)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
'''


# this was only for position need to get array of all directional trajectories including pos, vel,acc, and jerk
'''
x_traj_approach = [x_t(t) for t in time_approach]
y_traj_approach = [y_t(t) for t in time_approach]
z_traj_approach = [z_t(t) for t in time_approach]

x_traj_departure = [x_t1(t) for t in time_departure]
y_traj_departure = [y_t1(t) for t in time_departure]
z_traj_departure = [z_t1(t) for t in time_departure]
'''
# Generate trajectory data for both segments
x_traj_approach = [x_t(t) for t in time_approach]
y_traj_approach = [y_t(t) for t in time_approach]
z_traj_approach = [z_t(t) for t in time_approach]
vx_traj_approach = [vx_t(t) for t in time_approach]
vy_traj_approach = [vy_t(t) for t in time_approach]
vz_traj_approach = [vz_t(t) for t in time_approach]
ax_traj_approach = [ax_t(t) for t in time_approach]
ay_traj_approach = [ay_t(t) for t in time_approach]
az_traj_approach = [az_t(t) for t in time_approach]

x_traj_departure = [x_t1(t) for t in time_departure]
y_traj_departure = [y_t1(t) for t in time_departure]
z_traj_departure = [z_t1(t) for t in time_departure]
vx_traj_departure = [vx_t1(t) for t in time_departure]
vy_traj_departure = [vy_t1(t) for t in time_departure]
vz_traj_departure = [vz_t1(t) for t in time_departure]
ax_traj_departure = [ax_t1(t) for t in time_departure]
ay_traj_departure = [ay_t1(t) for t in time_departure]
az_traj_departure = [az_t1(t) for t in time_departure]




x_traj = np.concatenate((x_traj_approach, x_traj_departure))
y_traj = np.concatenate((y_traj_approach, y_traj_departure))
z_traj = np.concatenate((z_traj_approach, z_traj_departure))
vx_traj = np.concatenate((vx_traj_approach, vx_traj_departure))
vy_traj = np.concatenate((vy_traj_approach, vy_traj_departure))
vz_traj = np.concatenate((vz_traj_approach, vz_traj_departure))
ax_traj = np.concatenate((ax_traj_approach, ax_traj_departure))
ay_traj = np.concatenate((ay_traj_approach, ay_traj_departure))
az_traj = np.concatenate((az_traj_approach, az_traj_departure))

time_full = np.concatenate((time_approach, time_departure + tf))

# Columns: time, x, y, z, vx, vy, vz, ax, ay, az
traj = np.column_stack((time_full,
                        x_traj, y_traj, z_traj,
                        vx_traj, vy_traj, vz_traj,
                        ax_traj, ay_traj, az_traj))

print(traj)
'''
#use this in main code to combine all polynomials


# to obtain the correcet thrust orientation and aligned with gate, this is a wip
def compute_orientation_quaternion(ax, ay, az):
    thrust_vector = np.array([ax,ay,(az-9.81)])
    thrust_unit = thrust_vector / np.linalg.norm(thrust_vector)

    z_body = np.array([0,0,1])
    cross_prod = np.cross(z_body, thrust_unit)
    dot_prod = np.dot(z_body, thrust_unit)

    if np.allclose(cross_prod, 0):
        if dot_prod > 0:
            return np,array([0,0,0,1])
        else:
            return np.array([1,0,0,0]) 
            axis = cross_prod / np.linalg.norm(cross_prod)
            angle = np.arccos(np.clip(dot_prod, -1,1))
            return R.from_rotvec(angle*axis).as_quat()
def get_thrust_vector_and_quaternion(t):
    """Return thrust vector and orientation quaternion at time t."""
    if t <= tf:
        ax = ax_t(t)
        ay = ay_t(t)
        az = az_t(t)
    else:
        t1 = t - tf
        ax = ax_t1(t1)
        ay = ay_t1(t1)
        az = az_t1(t1)

    q = compute_orientation_quaternion(ax, ay, az)
    return np.array([ax, ay, (az - 9.81)]), q
thrust_vec, quat= get_thrust_vector_and_quaternion(tf)
print('Thrust vector:', thrust_vec)
print('Quaternion [x,y,z,w]', quat)

dt = 0.01
pos = []
vel = []
acc = []
q_traj = []
euler_ang = []
for t in np.linspace(0, tf + tf1, 200):
    if t <=tf:
        x = x_t(t)
        y = y_t(t)
        z = z_t(t)
        vx = vx_t(t)
        vy = vy_t(t)
        vz = vz_t(t)
        ax = ax_t(t)
        ay = ay_t(t)
        az = az_t(t)
    else: 
        x = x_t1(tf1)
        y = y_t1(tf1)
        z = z_t1(tf1)
        vx = vx_t1(tf1)
        vy = vy_t1(tf1)
        vz = vz_t1(tf1)
        ax = ax_t1(tf1)
        ay = ay_t1(tf1)
        az = az_t1(tf1)

pos.append([x,y,z])
vel.append([vx,vy,vz])
acc.append([ax,ay,az])
thrust_vec = np.array([-ax,-ay,-(az + 9.81)])
q = compute_orientation_quaternion(ax,ay,az)
q_traj.append(q)
'''

# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj, label='Drone Approach Trajectory')
#ax.plot(x_traj_departure, y_traj_departure, z_traj_departure, label='Drone Departure Trajectory')
ax.plot([0], [0], [1], 'ro', markersize=5, label='Gate Origin')  # gate at (0,0,1)
gate = np.array([
    [-0.5, 0, -0.25],
    [ 0.5, 0, -0.25],
    [ 0.5, 0,  0.25],
    [-0.5, 0,  0.25],
    [-0.5, 0, -0.25]  
])

# 45-degree rotation about Y-axis for gate
theta = np.radians(45)
ty = np.array([
    [ np.cos(theta), 0, np.sin(theta)],
    [ 0,             1, 0            ],
    [-np.sin(theta), 0, np.cos(theta)]
])

# Rotate and translate gate to origin at (0,0,1)
gate_pts = gate @ ty.T + np.array([0, 0, 1])
ax.plot(gate_pts[:, 0], gate_pts[:, 1], gate_pts[:, 2], 'g-', lw=2)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4]) 
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Drone Trajectory through Gate')
ax.legend()
ax.grid(True)
plt.show()

