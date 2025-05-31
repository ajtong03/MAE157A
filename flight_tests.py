import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import datetime
import pandas as pd

### Import custom modules and classes ###
from dynamics import dynamics
import trajectory
from PositionController import PositionController
from AttitudeController import AttitudeController
from quaternionfunc import *
from Updater import Updater

##########################################
############ Drone Simulation ############
##########################################

# Save data flag
save_data = False

# Initial conditions
t = 0.


# initialise starting position, velocity, quaternion, angular velocity
# these values are dependent on trajectory

# Final time from trajectory
tf = trajectory.total_time #10.
# print(tf)

# Simulation rate
dt = 1/100

# Gravity
g = 9.8
params = np.array([g])

# Initialize dynamics
dyn = dynamics(params, dt)
pos = PositionController(params, dt)
att = AttitudeController(params, dt)

# Initialize data array that contains useful info (probably should add more)
# CONFIGURE THIS

# initilize 3D plot
sim = Updater()
states = []
thrust_profile = []
motor_forces = []
time = []


#######################################################################################
# ------------------------------------ Actual Data ------------------------------------
####################################################################################### 
inner = pd.read_csv('Day 1/test1_inner_loop.csv')
outer = pd.read_csv('Day 1/test1_outer_loop.csv')

inner = inner.iloc[708:1000]
outer = outer.iloc[1427:2065]
#print('inner ', inner)
#print('outer', outer)

inner['int_time'] = (inner['t'] * 1000).astype('int64')
outer['int_time'] = (outer['t'] * 1000).astype('int64')

inner = inner.set_index('int_time')
outer = outer.set_index('int_time')

start = max(inner.index.min(), outer.index.min())
end = min(inner.index.max(), outer.index.max())
#print('first: ', outer.index.min(), ' last: ', outer.index.max())
#print('start: ', start, ' end: ', end)
#print('inner: ', inner.index.min(), ' last: ', inner.index.max())
step = 10  # ms

interp_time = np.arange(start, end, step)

inner_interp = inner.reindex(interp_time).interpolate(method='index', limit_direction = 'both').reset_index()
outer_interp = outer.reindex(interp_time).interpolate(method='index', limit_direction = 'both').reset_index()

print(outer_interp)
print(inner_interp)

inner = inner_interp
outer = outer_interp

time = (interp_time - start) / 1000
qw = inner['qw'].to_numpy()
qx = inner['qx'].to_numpy()
qy = inner['qy'].to_numpy()
qz = inner['qz'].to_numpy()

wx = inner['wx'].to_numpy()
wy = inner['wy'].to_numpy()
wz = inner['wz'].to_numpy()

x = outer['x'].to_numpy()
y= outer['y'].to_numpy()
z = outer['z'].to_numpy()

vx = outer['vx'].to_numpy()
vy = outer['vy'].to_numpy()
vz = outer['vz'].to_numpy()

## account for position offsets
x_off = -1.25 - x[0]
x = x + x_off
y_off = 1 - y[0]
y = y + y_off
z_off = 0.6 - z[0]
z = z + z_off

states = np.zeros((len(x), 13))
states[:, 0] = x
states[:, 1] = y
states[:, 2] = z
states[:, 3] = vx
states[:, 4] = vy
states[:, 5] = vz
states[:, 6] = qw
states[:, 7] = qx
states[:, 8] = qy
states[:, 9] = qz
#print (inner)
#print(outer)
#test_fig = plt.figure(5)
#plt.plot(t_o, x)

#######################################################################################
# -------------------------------------- Desired --------------------------------------
####################################################################################### 
# initialise trajectory
traj_start = trajectory.traj_State(0)
# print(traj, 'end')
p_start = traj_start[0:3]
#print('start', p_start)
v_start = traj_start[3:6]
q_start = [1.0, 0., 0., 0.]
w_start = [0., 0., 0.] # in radians

# initialise starting state
f = np.zeros(4)
state_cur = np.array(list(p_start) + list(v_start) + q_start + w_start)
des_states = []
for t in time:    # Get new desired state from trajectory planner
    xd, yd, zd, vx_d, vy_d, vz_d, ax_d, ay_d, az_d, jx_d, jy_d, jz_d = trajectory.traj_State(t)

    a_d = np.array([ax_d, ay_d, az_d])
    j_d = np.array([jx_d, jy_d, jz_d])

    target_state = np.zeros(13)
    target_state[0:3] = np.array([xd, yd, zd])
    target_state[3:6] = np.array([vx_d, vy_d, vz_d])

    q_d, w_d, thrust, a = pos.posController(state_cur, target_state, a_d, j_d)
    
    target_state[6:10] = q_d
    target_state[10:13] = w_d

    torque = att.attController(state_cur, target_state)
    f = att.getForces(torque, thrust)
    state_cur = dyn.propagate(state_cur, f)

    des_states.append(state_cur.copy())
    
    t += dt

des_states = np.array(des_states)
'''
# Plot velocity, thrust, and motor forces profiles
states = np.array(states)
thrust_profile = np.array(thrust_profile)
motor_forces = np.array(motor_forces)

vel_fig = plt.figure(2)
time = np.array(time)
plt.plot(time, states[:, 3], label = 'x velocity')
plt.plot(time, states[:, 4], label = 'y velocity')
plt.plot(time, states[:, 5], label = 'z velocity')
plt.title('Velocity Profile')
plt.legend()

thrust_fig = plt.figure(3)
plt.plot(time, thrust_profile, label = 'thrust magnitude')
plt.title('Thrust Profile')
plt.legend()

motor_fig = plt.figure(4)
plt.plot(time, motor_forces[:, 0], label = 'motor 1')
plt.plot(time, motor_forces[:, 1], label = 'motor 2')
plt.plot(time, motor_forces[:, 2], label = 'motor 3')
plt.plot(time, motor_forces[:, 3], label = 'motor 4')
plt.title('Motor Forces Profile')
plt.legend()
'''
pos = plt.figure(2)
plt.plot(time, x, label = 'actual x')
plt.plot(time, y, label = 'actual y')
plt.plot(time, z, label = 'actual z')
plt.plot(time, des_states[:, 0], label = 'desired x')
plt.plot(time, des_states[:, 1], label = 'desired y')
plt.plot(time, des_states[:, 2], label = 'desired z')
plt.title('Position Profile')
plt.legend()

# --- run animation ------------------------------------------------
sim.initializePlot()
anim_fig = sim.fig
def animate(i):
    if i < len(states):
        tail_artists = sim.updateTrail(states[i])
        drone_artists = sim.updateDrone(states[i], dyn)
        return tail_artists + drone_artists
    else:
        return []


sim.updateDrone(states[0], dyn)
ani = animation.FuncAnimation(anim_fig, animate, frames=len(states), interval=dt/100, blit=False, repeat = False)
plt.show()