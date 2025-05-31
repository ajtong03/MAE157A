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
rate = 500
dt = 1./rate

# Gravity
g = 9.8
params = np.array([g])

# Initialize dynamics
dyn = dynamics(params, dt)
pos = PositionController(params, dt)
att = AttitudeController(params, dt)

# Initialize data array that contains useful info (probably should add more)
# CONFIGURE THIS


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

# Initialize data array that contains useful info (probably should add more)
# CONFIGURE THIS
data = np.append(t,state_cur)
data = np.append(data,f)


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


qw = inner['qw'].to_numpy()
qx = inner['qx'].to_numpy()
qy = inner['qy'].to_numpy()
qz = inner['qz'].to_numpy()

wx = inner['wx'].to_numpy()
wy = inner['wy'].to_numpy()
wz = inner['wz'].to_numpy()

t_o = outer['t'].to_numpy()
t_o = t_o[750:]
#print('time:',  t_o[0], ', ', t_o[len(t_o) - 1])
t_o = t_o - t_o[0]
x = outer['x'].to_numpy()[750:]
y= outer['y'].to_numpy()[750:]
z = outer['z'].to_numpy()[750:]

vx = outer['vx'].to_numpy()[750:]
vy = outer['vy'].to_numpy()[750:]
vz = outer['vz'].to_numpy()[750:]

## account for position offsets
x_off = -1.25 - x[0]
x = x + x_off
y_off = 1 - y[0]
y = y + y_off
z_off = 0.15 - z[0]
z = z - z_off

states = np.zeros((len(x), 13))
states[:, 0] = x
states[:, 1] = y
states[:, 2] = z
states[:, 3] = vx
states[:, 4] = vy
states[:, 5] = vz
states[:, 6] = 1
states[:, 7] = 0
states[:, 8] = 0
states[:, 9] = 0
#print (inner)
#print(outer)
#test_fig = plt.figure(5)
#plt.plot(t_o, x)



# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"
    np.savetxt("../data/"+file_name, data, delimiter=",")

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