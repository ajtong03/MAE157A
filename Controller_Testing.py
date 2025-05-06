import numpy as np
import datetime
import matplotlib.pyplot as plt

### Import custom modules and classes ###
from dynamics import dynamics
import quaternionfunc
from AttitudeController import AttitudeController

##########################################
############ Drone Simulation ############
##########################################

# Save data flag
save_data = False

# Initial conditions
t = 0.

state = np.zeros(13)
f = np.zeros(4)

# x, y, z
state[0] = 2
state[1] = 2.3
state[2] = 2

# vx, vy, vz
state[3] = 0.
state[4] = 0.
state[5] = 0.

# 10 degrees off vertical
pitch = 10 * np.pi / 180
q = quaternionfunc.euler_to_quat(0, pitch, 0)
# qw, qx, qy, qz
state[6] = q[0]
state[7] = q[1]
state[8] = q[2]
state[9] = q[3]

# wx, wy, wz
state[10] = 0.
state[11] = 0.
state[12] = 0.

# Final time
tf = 10.

# Simulation rate
rate = 500
dt = 1./rate

# Gravity
g = 9.8

# Other parameters?
# m = ...

# Initialize dynamics
dyn = dynamics(np.array([g]), dt)
att = AttitudeController(np.array([g]), dt)

# Initialize data array that contains useful info (probably should add more)
data = np.append(t,state)
data = np.append(data,f)


# TEST ATTITUDE CONTROLLER
target_state = np.zeros(13)

target_state[0] = 2
target_state[1] = 2
target_state[2] = 2

target_state[6] = 1.
target_state[7] = 0.
target_state[8] = 0.
target_state[9] = 0.

#Kp = np.diag([3, 3, 3])
#Kd = np.diag([2, 2, 2])
#gains = np.array[Kp, Kd]

q_d = target_state[6:10]

err = quaternionfunc.error(state[6:10], q_d)
error_data = np.append(t, err)

#Kp, Kd = AttitudeController.setAttController2(att, state, q_d, np.array([9.81*dyn.l]))
#print(Kd)
#print(Kp)
# Simulation loop

running = True
while running:

    # Propagate dynamics with control inputs
    Kp = np.diag([2.5, 2.5, 2.5])
    Kd = np.diag([5, 5, 5])
    #gains = np.array[Kp, Kd]

    torque = AttitudeController.attController(att, state, target_state, Kp, Kd)
    # set thrust so z component equals gravity
    thrust = att.m * 9.81 / pitch
    #print(thrust)
    f = AttitudeController.getForces(att, Kp, Kd, torque, np.array([20]))
    state = dyn.propagate(state, f, dt)
    q_e = quaternionfunc.error(state[6:10], q_d)
    # If z too low then indicate crash and end simulation
    if state[2] < 0.1:
        print("CRASH!!!")
        break

    # Update data array (this can probably be done in a much cleaner way...)
    #tmp = np.append(tmp,f)
    #tmp = np.append(t,state)
    #data = np.vstack((data,tmp))

    error = np.append(t, q_e)
    error_data = np.vstack((error_data,error))

    # Update time
    t += dt 

    # If time exceeds final time then stop simulator
    if t >= tf:
        running = False

# plot the error to see how well the gains converge
time = error_data[:,0]
q_w = error_data[:, 1]
q_x = error_data[:, 2]
q_y = error_data[:, 3]
q_z = error_data[:, 4]

plt.plot(time, q_w, label = 'q_w')
plt.plot(time, q_x, label = 'q_x')
plt.plot(time, q_y, label = 'q_y')
plt.plot(time, q_z, label = 'q_z')
plt.legend()
plt.grid(True)
plt.show()


# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"
    np.savetxt("../data/"+file_name, data, delimiter=",")




