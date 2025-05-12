import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import datetime

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
time = []
# Simulation loop

t = 0.0
running = True
while running:
    # Get new desired state from trajectory planner
    xd, yd, zd, vx_d, vy_d, vz_d, ax_d, ay_d, az_d, jx_d, jy_d, jz_d = trajectory.traj_State(t)

    time.append(t)
    # print('vel: ', vx_d, ' ', vy_d, ' ', vz_d)
    a_d = np.array([ax_d, ay_d, az_d])
    # p_d = np.array([xd, yd, zd])
    # print(p_d)
    j_d = np.array([jx_d, jy_d, jz_d])

    target_state = np.zeros(13)
    target_state[0:3] = np.array([xd, yd, zd])
    target_state[3:6] = np.array([vx_d, vy_d, vz_d])
    print('target ', target_state[0:3])

    q_d, w_d, thrust = pos.posController(state_cur, target_state, a_d, j_d)

    thrust_profile.append(thrust.copy())

    target_state[6:10] = q_d
    target_state[10:13] = w_d

    torque = att.attController(state_cur, target_state)
    f = att.getForces(torque, thrust)
    state_cur = dyn.propagate(state_cur, f)
    print('current ', state_cur[0:3])

    states.append(state_cur.copy())
    # If z to low then indicate crash and end simulation
    # t > 2 set arbitrarily so it doesn't "CRASH" at the start
    if state_cur[2] < 0.1:
        if t > 2:
            print("CRASH!!!")
            break

    t += dt
    if t >= tf:
        # break if the end of the trajectory has been reached
        running = False
        print('End of trajectory reached')
        break
    

    # Update data array (this can probably be done in a much cleaner way...)
    tmp = np.append(t,state_cur)
    tmp = np.append(tmp,f)
    data = np.vstack((data,tmp))

    sim.updatePlot(state_cur, dyn)

# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"
    np.savetxt("../data/"+file_name, data, delimiter=",")


# --- run animation ------------------------------------------------
sim.initializePlot()
anim_fig = plt.figure()
def animate(i):
    if i < len(states):
        return sim.updatePlot(states[i], dyn)
#frames = int((trajectory.tf + trajectory.tf1)/dt) + 1
ani = animation.FuncAnimation(anim_fig, animate, frames=len(states), interval=dt*1000, blit=False, repeat = False)

# ---------- plot thrust and velocity profiles ----------

states = np.array(states)
thrust_profile = np.array(thrust_profile)

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

plt.show()

