import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sin,cos, tan
from dynamics import dynamics   # import the class, not the module
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from PositionController import PositionController
# 1m in y -dir, y acc = -8.25, 4.94°
#45° gate

# during OH with the prof, the prof recommend having a neg accel in y-dir (but no velocity in y-dir) 
# and velocity in x-direction (but no accel in x-dir)
# no accel in z-dir either
#trajectories 7+all have y-dir accel, prev ones involved x-dir accel

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
# since T = m*g our total thrust must be > 8.30907 N to lift
# Approach Segment 
# Recommended to have 0 acceleration through the gate and thrust perpendicular to gate's side
# x - axis
tf = 1.6 # seconds
time_approach = np.linspace(0, tf, 200)

c_x = solve_polynomial_coefficients(tf, -1.25, 0, 0, 0, 0, -3, 0,0)
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
# remember acceleration <==> orientation/thrust (yaw in z-dir rotates plane left and right)
c_y = solve_polynomial_coefficients(tf,0.5, 0, 0, 0, -1, 0, 18, 0)
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
c_z = solve_polynomial_coefficients(tf, 0.9, 0, 0, 0, 1.5, 0, 0, 0 )
print("z-coeffs:", c_z)
# needs to counteract gravity in this case
def z_t(t):
    return c_z[0] + c_z[1] * t + c_z[2] * t**2 + c_z[3] * t**3 + c_z[4] * t**4 + c_z[5] * t**5 + c_z[6] * t**6 + c_z[7] * t**7
def vz_t(t):
    return c_z[1] + 2 * c_z[2] * t + 3 * c_z[3] * t**2 + 4 * c_z[4] * t**3 + 5 * c_z[5] * t**4 + 6 * c_z[6] * t**5 + 7 * c_z[7] * t**6
def az_t(t):
    return 2 * c_z[2] + 6 * c_z[3] * t + 12 * c_z[4] * t**2 + 20 * c_z[ 5] * t**3 + 30 * c_z[6] * t**4 + 42 * c_z[7] * t**5 
def jz_t(t):
    return 6 * c_z[3] + 24 * c_z[4] * t + 60 * c_z[5] * t**2 + 120 * c_z[6] * t**3 + 210 * c_z[7] * t**4


def traj_State(t):
    x = x_t(t)
    y = y_t(t)
    z = z_t(t)

    vx = vx_t(t)
    vy = vy_t(t)
    vz = vz_t(t)

    ax = ax_t(t)
    ay = ay_t(t)
    az = az_t(t)

    jx = jx_t(t)
    jy = jy_t(t)
    jz = jz_t(t)

    state = np.array([x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz])

    return state


# Departure Segment 
# Initial Boundary Conditions must match final BCs from approach segment
tf1 =  1.2 # seconds
time_departure = np.linspace(0, tf1, 200)

# x - axis
c_x1 = solve_polynomial_coefficients(tf1, 0, -3, 0, 0, -1, 0, 0, 0 )
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
c_y1 = solve_polynomial_coefficients(tf1, -1 , 0, 18, 0, -1.25, 0, 0, 0 )
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
#during testing change z-altitude based on how it lands
c_z1 = solve_polynomial_coefficients(tf1, 1.5, 0, 0, 0, 0.6, 0, 0, 0)
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


# ------------------------------------------ RETURN STATE AT ANY TIME t FOR TRAJECTORY ------------------------------------------- 
# -------------- given a time t, return the position, velocity, acceleration, and jerk of the trajectory at that point -----------
def traj_State(t):
    
    if t <= tf:
        x = x_t(t)
        y = y_t(t)
        z = z_t(t)

        vx = vx_t(t)
        vy = vy_t(t)
        vz = vz_t(t)

        ax = ax_t(t)
        ay = ay_t(t)
        az = az_t(t)

        jx = jx_t(t)
        jy = jy_t(t)
        jz = jz_t(t)
    else:
        t = t - tf
        x = x_t1(t)
        y = y_t1(t)
        z = z_t1(t)

        vx = vx_t1(t)
        vy = vy_t1(t)
        vz = vz_t1(t)

        ax = ax_t1(t)
        ay = ay_t1(t)
        az = az_t1(t)

        jx = jx_t1(t)
        jy = jy_t1(t)
        jz = jz_t1(t)
    state = np.array([x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz])

    return state

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
jx_traj_approach = [jx_t(t) for t in time_approach]
jy_traj_approach = [jy_t(t) for t in time_approach]
jz_traj_approach = [jz_t(t) for t in time_approach]

x_traj_departure = [x_t1(t) for t in time_departure]
y_traj_departure = [y_t1(t) for t in time_departure]
z_traj_departure = [z_t1(t) for t in time_departure]
vx_traj_departure = [vx_t1(t) for t in time_departure]
vy_traj_departure = [vy_t1(t) for t in time_departure]
vz_traj_departure = [vz_t1(t) for t in time_departure]
ax_traj_departure = [ax_t1(t) for t in time_departure]
ay_traj_departure = [ay_t1(t) for t in time_departure]
az_traj_departure = [az_t1(t) for t in time_departure]
jx_traj_departure = [jx_t1(t) for t in time_departure]
jy_traj_departure = [jy_t1(t) for t in time_departure]
jz_traj_departure = [jz_t1(t) for t in time_departure]

x_traj = np.concatenate((x_traj_approach, x_traj_departure))
y_traj = np.concatenate((y_traj_approach, y_traj_departure))
z_traj = np.concatenate((z_traj_approach, z_traj_departure))
vx_traj = np.concatenate((vx_traj_approach, vx_traj_departure))
vy_traj = np.concatenate((vy_traj_approach, vy_traj_departure))
vz_traj = np.concatenate((vz_traj_approach, vz_traj_departure))
ax_traj = np.concatenate((ax_traj_approach, ax_traj_departure))
ay_traj = np.concatenate((ay_traj_approach, ay_traj_departure))
az_traj = np.concatenate((az_traj_approach, az_traj_departure))
jx_traj = np.concatenate((jx_traj_approach, jx_traj_departure))
jy_traj = np.concatenate((jy_traj_approach, jy_traj_departure))
jz_traj = np.concatenate((jz_traj_approach, jz_traj_departure))

time_full = np.concatenate((time_approach, time_departure + tf))
total_time = tf + tf1
# Columns: time, x, y, z, vx, vy, vz, ax, ay, az
traj = np.column_stack((time_full,
                        x_traj, y_traj, z_traj,
                        vx_traj, vy_traj, vz_traj,
                        ax_traj, ay_traj, az_traj,
                        jx_traj, jy_traj, jz_traj))
# Separately plot polynomials (best to see distance it covers)
# change as necessary
# position

plt.plot(time_full, x_traj, label="xPos")
plt.plot(time_full, y_traj, label="yPos")
plt.plot(time_full, z_traj, label="ZPos")
plt.tight_layout()
plt.xlabel('time')
plt.ylabel("distance")
plt.title("distance covered")
plt.legend()


# velocity 
plt.figure(figsize=(8, 6))
plt.plot(time_full, vx_traj, label="xvel")
plt.plot(time_full, vy_traj, label="yvel")
plt.plot(time_full, vz_traj, label="zvel")

#plt.plot(time_values, [jx_t(t) for t in time_departure], label = "Jerk")
plt.title("X-Direction Trajectory (app segment)")
plt.xlabel("Time (s)")
plt.ylabel("Vel")
plt.title("Velocity vs. Time")
plt.tight_layout()
plt.legend()
plt.grid(True)


# acceleration of the full trajectory

plt.figure(figsize=(10, 6))
plt.plot(time_full, ax_traj, label='ax (m/s²)')
plt.plot(time_full, ay_traj, label='ay (m/s²)')
plt.plot(time_full, az_traj, label='az (m/s²)')

plt.xlabel('t')
plt.ylabel('accel')
plt.title("Acceleration vs. Time")
plt.grid(True)
plt.legend()
plt.tight_layout()


# print(traj)
v_mag = np.sqrt(vx_traj**2 + vy_traj**2 + vz_traj**2)
# print(f'Velocity magnitude: ', v_mag)

# total thrust 
m = 0.847 
g = 9.81

T_vector = m* np.vstack( np.sqrt(ax_traj**2 + ay_traj**2 + (az_traj+g)**2)) # to use for thrust at gate
T_mag = m* np.sqrt(ax_traj**2 + ay_traj**2 + (az_traj+g)**2) # to check feasibility
#a_yd = 
#print(T_vector)
print('Thrust Mag: ', T_mag)


# to obtain the correct thrust orientation and aligned with gate
def compute_orientation_quaternion(ax, ay, az):
    thrust_vector = np.array([ax,ay,(az+9.81)])
    thrust_unit = thrust_vector / np.linalg.norm(thrust_vector)

    z_body = np.array([0,0,1])
    cross_prod = np.cross(z_body, thrust_unit)
    dot_prod = np.dot(z_body, thrust_unit)

    if np.allclose(cross_prod, 0):
        if dot_prod > 0:
            return np.array([0,0,0,1])
        else:
            return np.array([1,0,0,0]) 
            axis = cross_prod / np.linalg.norm(cross_prod)
            angle = np.arccos(np.clip(dot_prod, -1,1))
            return R.from_rotvec(angle*axis).as_quat()
# to check if trajectory is feasible
params = [g]
dt = time_full[1] - time_full[0]
position_controller = PositionController(params,dt)
T_min = position_controller.minThrust
T_max = position_controller.maxThrust

feasible = True
for i, thrust in enumerate(T_mag):
    if thrust > T_max or thrust < T_min:
        feasible = False
        print(f"Trajectory is NOT feasible at t = {time_full[i]:.2f} s. Thrust: {T_mag[i]:.2f}")
        break

if feasible:
    print("Trajectory is feasible over the entire duration.")

# 3D Plot of trajectory
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj, y_traj, z_traj, color='blue',label='Drone Trajectory')

# wip magnitudes of acceleration
skip =5
ax.quiver(
    x_traj[::skip], y_traj[::skip], z_traj[::skip],
    ax_traj[::skip],ay_traj[::skip], az_traj[::skip],
    length=0.2, normalize=True, color='r', label='acceleration vectors')
# Gate
ax.plot([0], [-1], [1.5], 'ro', markersize=5, label='Gate Origin')  
gate = np.array([
            [0, -0.25, -0.1905], # bottom left
            [0,  0.25, -0.1905], # bottom right
            [0,  0.25,  0.1905], # top right
            [0, -0.25,  0.1905], # top left
            [0, -0.25, -0.1905]  # close
        ])


# 45-degree rotation about Y-axis for gate
theta = np.radians(60) # (-) --> CCW (+) --> CW
ty = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)           ],
            [0, -np.sin(theta), np.cos(theta)]
        ])


# Rotate and translate gate to origin at (0,0,1)
initial_normal = np.array([0, 0, 1])  # Normal 
gate_pts = gate @ ty.T + np.array([0, -1, 1.5])
gate_normal = ty @ initial_normal  # Rotate the normal, not the origin
ax.plot(gate_pts[:, 0], gate_pts[:, 1], gate_pts[:, 2], color = 'black', lw=2)
ax.quiver(
    0, -1, 1.5,                    
    gate_normal[0], gate_normal[1], gate_normal[2],  # Components

    length=0.5, color='purple', linewidth=2, label='Gate Normal'
)

# to double check thrust vector aligns with normal gate 
t_gate = tf  # for calculating time when drone is at the gate
gate_idx = np.argmin(np.abs(time_full - t_gate))
a_gate =  m* np.array([
    ax_traj[gate_idx],
    ay_traj[gate_idx],
    az_traj[gate_idx] + g  
])
thrust_unit = a_gate / np.linalg.norm(a_gate)
T_mag_gate = np.linalg.norm(a_gate)
# thrust vector at gate
ax.quiver(
    0, -1, 1.5,  
    thrust_unit[0], thrust_unit[1], thrust_unit[2],
    length=0.75, color='orange', linewidth=2, label='Thrust at Gate'
)
# check alignment <10 degrees is good
angle_rad = np.arccos(np.clip(np.dot(thrust_unit, gate_normal) /
                              (np.linalg.norm(thrust_unit) * np.linalg.norm(gate_normal)), -1, 1))
angle_deg = np.degrees(angle_rad)
print(f"Angle between gate normal and thrust vector: {angle_deg:.2f}°, good if < 10°")

#print(f"thrust vector at gate: {thrust_vector_at_gate}")
print(f"Normalized Gate Normal: {gate_normal}")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4]) 
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f"3D Drone Trajectory through Gate, Thrust at Gate = {float(T_mag_gate):.2f} N")
ax.legend()
ax.grid(True)
plt.show()

