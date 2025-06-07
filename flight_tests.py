import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import savgol_filter
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


# initilize 3D plot
sim = Updater()
states = []
thrust_profile = []
motor_forces = []
time = []


#######################################################################################
# ------------------------------------ Actual Data ---------------------------------- #
####################################################################################### 
inner = pd.read_csv('Day 2/test3_inner_loop.csv')
outer = pd.read_csv('Day 2/test3_outer_loop.csv')

# test 1
#inner = inner.iloc[713:990]
#outer = outer.iloc[1437:2065]

# test 2
#outer = outer.iloc[2401:3074]
#inner = inner.iloc[1196:1532]

# test 3
#outer = outer.iloc[2350:3100]
#inner = inner.iloc[1180:1520]

# test 4
#inner = inner.iloc[1040:1270]
#outer = outer.iloc[2100:2600]

# test 5
#inner = inner.iloc[820:1060]
#outer = outer.iloc[1637:2165]

# test 6
inner = inner.iloc[1220:1440]
outer = outer.iloc[2437:2865]

inner['int_time'] = (inner['t'] * 1000).astype('int64')
outer['int_time'] = (outer['t'] * 1000).astype('int64')

inner = inner.set_index('int_time')
outer = outer.set_index('int_time')

start = max(inner.index.min(), outer.index.min())
end = min(inner.index.max(), outer.index.max())
#print('start: ', start, ' end: ', end)
#print('outer: ', outer.index.min(), ' to ', outer.index.max())
#print('inner: ', inner.index.min(), ' to ', inner.index.max())
step = 10  # ms

interp_time = np.arange(start, end, step)

inner_interp = inner.reindex(interp_time).interpolate(method='index', limit_direction = 'both').reset_index()
outer_interp = outer.reindex(interp_time).interpolate(method='index', limit_direction = 'both').reset_index()

##print(outer_interp)
print(inner_interp)

#inner = inner_interp
#outer = outer_interp

time_o = outer['t'].to_numpy()
time_o = time_o - time_o[0]
time_i = inner['t'].to_numpy()
time_i = time_i - time_i[0]
#time = (interp_time - start) / 1000

# --------- Actual States --------- #
qw = inner['qw'].to_numpy()
qx = inner['qx'].to_numpy()
qy = inner['qy'].to_numpy()
qz = inner['qz'].to_numpy()

wx = inner['wx'].to_numpy()
wx = savgol_filter(wx, 31, 3)
wy = inner['wy'].to_numpy()
wy = savgol_filter(wy, 31, 3)
wz = inner['wz'].to_numpy()
wz = savgol_filter(wz, 31, 3)


x = outer['x'].to_numpy()
y = outer['y'].to_numpy()
z = outer['z'].to_numpy()

vx = outer['vx'].to_numpy()
vx = savgol_filter(vx, 31, 3)
vy = outer['vy'].to_numpy()
vy = savgol_filter(vy, 31, 3)
vz = outer['vz'].to_numpy()
vz = savgol_filter(vz, 31, 3)

# --------- Desired States --------- #
xd = outer['xd'].to_numpy()
yd = outer['yd'].to_numpy()
zd = outer['zd'].to_numpy()
vxd = outer['vxd'].to_numpy()
vyd = outer['vyd'].to_numpy()
vzd = outer['vzd'].to_numpy()

qwd = inner['qwd'].to_numpy()
qxd = inner['qxd'].to_numpy()
qyd = inner['qyd'].to_numpy()
qzd = inner['qzd'].to_numpy()

wxd = inner['wxd'].to_numpy()
wxd = savgol_filter(wxd, 31, 3)
wyd = inner['wyd'].to_numpy()
wyd = savgol_filter(wyd, 31, 3)
wzd = inner['wzd'].to_numpy()
#wzd = savgol_filter(wzd, 31, 3)



# Creating Combined States
# states_p = np.zeros((len(outer), 6))
# states_a = np.zeros((len(inner), 7)) 
# states_p[:, 0] = outer['x'].to_numpy()
# states_p[:, 1] = outer['y'].to_numpy()
# states_p[:, 2] = outer['z'].to_numpy()
# states_p[:, 4] = vy
# states_p[:, 5] = vz

# states_a[:, 0] = inner['qw'].to_numpy()
# states_a[:, 1] = inner['qx'].to_numpy()
# states_a[:, 2] = inner['qy'].to_numpy()
# states_a[:, 3] = inner['qz'].to_numpy()
# states_a[:, 4] = inner['wx'].to_numpy()
# states_a[:, 5] = inner['wy'].to_numpy()
# states_a[:, 6] = inner['wz'].to_numpy()


# des_states_p = np.zeros((len(outer), 6))
# des_states_a = np.zeros((len(inner), 7))
# des_states_p[:, 0] = outer['xd'].to_numpy()
# des_states_p[:, 1] = outer['yd'].to_numpy()
# des_states_p[:, 2] = outer['zd'].to_numpy()
# des_states_p[:, 3] = outer['vxd'].to_numpy()
# des_states_p[:, 4] = outer['vyd'].to_numpy()
# des_states_p[:, 5] = outer['vzd'].to_numpy()

# des_states_a[:, 0] = inner['qwd'].to_numpy()
# des_states_a[:, 1] = inner['qxd'].to_numpy()
# des_states_a[:, 2] = inner['qyd'].to_numpy()
# des_states_a[:, 3] = inner['qzd'].to_numpy()
# des_states_a[:, 4] = inner['wxd'].to_numpy()
# des_states_a[:, 5] = inner['wyd'].to_numpy()
# des_states_a[:, 6] = inner['wzd'].to_numpy()


pos1 = plt.figure(2)
plt.plot(time_o, x, label = 'actual x')
plt.plot(time_o, y, label = 'actual y')
plt.plot(time_o, z, label = 'actual z')
plt.plot(time_o, xd, label = 'desired x')
plt.plot(time_o, yd, label = 'desired y')
plt.plot(time_o, zd, label = 'desired z')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Test 6 Position Profile')
plt.legend(bbox_to_anchor = (1, 1))

vel1 = plt.figure(3)
plt.plot(time_o, vx, label = 'actual vx')
plt.plot(time_o, vy, label = 'actual vy')
plt.plot(time_o, vz, label = 'actual vy')
plt.plot(time_o, vxd, label = 'desired vx')
plt.plot(time_o, vyd, label = 'desired vy')
plt.plot(time_o, vzd, label = 'desired vz')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Test 6 Velocity Profile')
plt.legend(bbox_to_anchor = (1, 1))


att1 = plt.figure(4)
plt.plot(time_i, qw, label = 'actual qw')
plt.plot(time_i, qx, label = 'actual qx')
plt.plot(time_i, qy, label = 'actual qy')
plt.plot(time_i, qz, label = 'actual qz')
plt.plot(time_i, qwd, label = 'desired qw')
plt.plot(time_i, qxd, label = 'desired qx')
plt.plot(time_i, qyd, label = 'desired qy')
plt.plot(time_i, qzd, label = 'desired qz')
plt.xlabel('Time (s)')
plt.ylabel('Attitude')
plt.title('Test 6 Orientation Profile')
plt.legend(bbox_to_anchor = (1, 1))

ang_vel = plt.figure(5)
plt.plot(time_i, wx, label = 'actual wx')
plt.plot(time_i, wy, label = 'actual wy')
plt.plot(time_i, wz, label = 'actual wz')
plt.plot(time_i, wxd, label = 'desired wx')
plt.plot(time_i, wyd, label = 'desired wy')
plt.plot(time_i, wzd, label = 'desired wz')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Test 6 Angular Velocity Profile')
plt.legend(bbox_to_anchor = (1, 1))

plt.show()

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
'''