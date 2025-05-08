# Filename: main.py
# Author: ...
# Created: ...
# Description: Drone simulator

### Import python packages ###
import numpy as np
import datetime

### Import custom modules and classes ###
from dynamics import dynamics
import trajectory
from PositionController import PositionController
from AttitudeController import AttitudeController
from quaternionfunc import *

##########################################
############ Drone Simulation ############
##########################################

# Save data flag
save_data = False

# Initial conditions
t = 0.

state = np.zeros(13)
f = np.zeros(4)

# initialise starting position, velocity, quaternion, angular velocity
# these values are dependent on trajectory
p_start = [0., 0., 0.]
v_start = [0.0, 0.0, 0.0]
q_start = [1.0, 0., 0., 0.]
w_start = [0., 0., 0.] # in radians

# initialise starting state
state_cur = np.array(p_start + v_start + q_start + w_start)

# Final time from trajectory
tf = trajectory.tf #10.

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
data = np.append(t,state)
data = np.append(data,f)

# initialise trajectory
traj = trajectory.traj

# Simulation loop
running = True
i = 0
while i in range(len(traj)):
    # Get new desired state from trajectory planner
    t, xd, yd, zd, vx_d, vy_d, vz_d, ax_d, ay_d, az_d, jx_d, jy_d, jz_d = traj[i]

    a_d = np.array([ax_d, ay_d, az_d])
    j_d = np.array([jx_d, jy_d, jz_d])

    target_state = np.zeros(13)
    target_state[0:3] = np.array([xd, yd, zd])
    target_state[3:6] = np.array([vx_d, vy_d, vz_d])

    q_d, w_d, thrust = pos.posController(state, target_state, a_d, j_d)

    target_state[6:10] = q_d
    target_state[10:13] = w_d

    torque = att.attController(state, target_state)
    f = att.getForces(torque, thrust)
    state_cur = dyn.propagate(state, f)

 
    # If z to low then indicate crash and end simulation
    if state[2] < 0.1:
        print("CRASH!!!")
        break
    elif t == tf:
        # break if the end of the trajectory has been reached
        print('End of trajectory reached')
        break

    # Update data array (this can probably be done in a much cleaner way...)
    tmp = np.append(t,state)
    tmp = np.append(tmp,f)
    data = np.vstack((data,tmp))

    # Update time
    t += dt 

    # If time exceeds final time then stop simulator
    if t >= tf:
        running = False

# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"
    np.savetxt("../data/"+file_name, data, delimiter=",")




