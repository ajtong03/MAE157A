import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# mass_drone = 0.399 kg

# x - dir
tf = 12  # seconds
c_x = solve_polynomial_coefficients(tf, -1.5, 0, 0, 0, 0, 2, 1.1, 0 )
print("x-coeffs:", c_x)
def x_t(t):
    return c_x[0] + c_x[1] * t + c_x[2] * t**2 + c_x[3] * t**3 + c_x[4] * t**4 + c_x[5] * t**5 + c_x[6] * t**6 + c_x[7] * t**7
def vx_t(t):
    return c_x[1] + 2 * c_x[2] * t + 3 * c_x[3] * t**2 + 4 * c_x[4] * t**3 + 5 * c_x[5] * t**4 + 6 * c_x[6] * t**5 + 7 * c_x[7] * t**6

def ax_t(t):
    return 2 * c_x[2] + 6 * c_x[3] * t + 12 * c_x[4] * t**2 + 20 * c_x[5] * t**3 + 30 * c_x[6] * t**4 + 42 * c_x[7] * t**5 

def jx_t(t):
    return 6 * c_x[3] + 24 * c_x[4] * t + 60 * c_x[5] * t**2 + 120 * c_x[6] * t**3 + 210 * c_x[7] * t**4
# Departure
tf1= 15 # seconds for departure segment only


'''
# Plotting
time_values = np.linspace(0, tf, 200)
# x - dir
plt.figure(figsize=(8, 6))
plt.plot(time_values, [x_t(t) for t in time_values], label="Position")
plt.plot(time_values, [vx_t(t) for t in time_values], label="Velocity")
plt.plot(time_values, [ax_t(t) for t in time_values], label="Acceleration")
plt.plot(time_values, [jx_t(t) for t in time_values], label = "Jerk")
plt.title("X-Direction Trajectory (approach segment)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# y - dir
c_y = solve_polynomial_coefficients(tf, -1.5 , 0 ,0, 0, 0, 0.2, -0.3, 0 )
print("y-coeffs:", c_y)
def y_t(t):
    return c_y[0] + c_y[1] * t + c_y[2] * t**2 + c_y[3] * t**3 + c_y[4] * t**4 + c_y[5] * t**5 + c_y[6] * t**6 + c_y[7] * t**7
def vy_t(t):
    return c_y[1] + 2 * c_y[2] * t + 3 * c_y[3] * t**2 + 4 * c_y[4] * t**3 + 5 * c_y[5] * t**4 + 6 * c_y[6] * t**5 + 7 * c_y[7] * t**6

def ay_t(t):
    return 2 * c_y[2] + 6 * c_y[3] * t + 12 * c_y[4] * t**2 + 20 * c_y[ 5] * t**3 + 30 * c_x[6] * t**4 + 42 * c_x[7] * t**5 

def jy_t(t):
    return 6 * c_y[3] + 24 * c_y[4] * t + 60 * c_y[5] * t**2 + 120 * c_y[6] * t**3 + 210 * c_x[7] * t**4


# Plotting
time_values = np.linspace(0, tf, 200)
plt.figure(figsize=(8, 6))
plt.plot(time_values, [y_t(t) for t in time_values], label="Position")
plt.plot(time_values, [vy_t(t) for t in time_values], label="Velocity")
plt.plot(time_values, [ay_t(t) for t in time_values], label="Acceleration")
plt.title("Y-Direction Trajectory (approach segment)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# z - dir 
c_z = solve_polynomial_coefficients(tf, 0, 0 ,0, 0, 1, 0, 0, 0 )
print("z-coeffs:", c_z)
# remember acceleration <==> orientation/thrust (yaw in z-dir rotates plane left and right)
# needs to counteract gravity in this case
def z_t(t):
    return c_z[0] + c_z[1] * t + c_z[2] * t**2 + c_z[3] * t**3 + c_z[4] * t**4 + c_z[5] * t**5 + c_z[6] * t**6 + c_z[7] * t**7
def vz_t(t):
    return c_z[1] + 2 * c_z[2] * t + 3 * c_z[3] * t**2 + 4 * c_z[4] * t**3 + 5 * c_z[5] * t**4 + 6 * c_z[6] * t**5 + 7 * c_z[7] * t**6

def az_t(t):
    return 2 * c_z[2] + 6 * c_z[3] * t + 12 * c_z[4] * t**2 + 20 * c_z[ 5] * t**3 + 30 * c_x[6] * t**4 + 42 * c_z[7] * t**5 

def jz_t(t):
    return 6 * c_z[3] + 24 * c_z[4] * t + 60 * c_z[5] * t**2 + 120 * c_z[6] * t**3 + 210 * c_z[7] * t**4

'''
# Plotting
time_values = np.linspace(0, tf, 200)

plt.figure(figsize=(8, 6))
plt.plot(time_values, [z_t(t) for t in time_values], label="Position")
plt.plot(time_values, [vz_t(t) for t in time_values], label="Velocity")
plt.plot(time_values, [az_t(t) for t in time_values], label="Acceleration")
plt.title("Z-Direction Trajectory (approach segment)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
'''
# Departure Segement

#[..]


# plotting in 3D
x_traj_approach = [x_t(t) for t in time_values]
y_traj_approach = [y_t(t) for t in time_values]
z_traj_approach = [z_t(t) for t in time_values]

# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj_approach, y_traj_approach, z_traj_approach, label='Drone Approach Trajectory')
ax.plot([0], [0], [1], 'ro', markersize=5, label='Gate Location')  # gate at (0,0,1)
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

