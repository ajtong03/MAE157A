import numpy as np
import datetime
import matplotlib.pyplot as plt

### Import custom modules and classes ###
from dynamics import dynamics
import quaternionfunc
from AttitudeController import AttitudeController
from PositionController import PositionController

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
state[0] = 0
state[1] = 0
state[2] = 0

# vx, vy, vz
state[3] = 0.
state[4] = 0.
state[5] = 0.

# 10 degrees off vertical
# pitch = 10 * np.pi / 180
#q = quaternionfunc.euler_to_quat(0, pitch, 0)
# qw, qx, qy, qz
'''
state[6] = q[0]
state[7] = q[1]
state[8] = q[2]
state[9] = q[3]
'''
state[6] = 1
state[7] = 0
state[8] = 0
state[9] = 0

# wx, wy, wz
state[10] = 0.
state[11] = 0.
state[12] = 0.

# Final time
tf = 0.4

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
pos = PositionController(np.array([g]), dt)

# Initialize data array that contains useful info (probably should add more)
data = np.append(t,state)
data = np.append(data,f)


############################################################################################################
#-----------------------------------------TEST ATTITUDE CONTROLLER-----------------------------------------#
# Set target state q_d to be the unit quaternion. Run attitude controller to see how well error converges
############################################################################################################

target_state = np.zeros(13)

target_state[0] = 2
target_state[1] = 2
target_state[2] = 2

target_state[6] = 2
target_state[7] = 1
target_state[8] = 1
target_state[9] = 1

#Kp = np.diag([3, 3, 3])
#Kd = np.diag([2, 2, 2])
#gains = np.array[Kp, Kd]

q_d = target_state[6:10]

err = quaternionfunc.error(state[6:10], q_d)
error_data = np.append(t, err)


#Kp, Kd = att.setAttController2(state, q_d)
#print(Kp)
#print(Kd)

# Simulation loop
runningA = True
testingA = runningA
while runningA:
   # print(t)
    # Propagate dynamics with control inputs
    Kp = np.diag([5.5, 5.5, 4.96])
    Kd = np.diag([.13, .146, .25])
    #Kp = np.diag([19, 18, 20])
    #Kd = np.diag([1.5, 1.5, 1.5])
    #Kp = np.diag([5, 5, 5])
    #Kd = np.diag([2.5, 2, 2.5])

    torque = att.attController_test(state, target_state, Kp, Kd)
    # set thrust so z component equals gravity
    thrust = 9.81 * att.m * 1.5
    f = att.getForces(torque, thrust)
    new_state = dyn.propagate(state, f, dt)
    q_e = quaternionfunc.error(new_state[6:10], q_d)
    
    state = new_state

    error = np.append(t, q_e)
    error_data = np.vstack((error_data,error))

    # Update time
    t += dt 

    # If time exceeds final time then stop simulator
    if t >= tf:
        runningA = False

if testingA == True:
    # plot the error to see how well the gains converge
    time = error_data[:,0]
    q_w = error_data[:, 1]
    q_x = error_data[:, 2]
    q_y = error_data[:, 3]
    q_z = error_data[:, 4]

    plt.figure(1)
    #plt.plot(time, q_w, label = 'q_w')
    plt.plot(time, q_x, label = 'q_x')
    plt.plot(time, q_y, label = 'q_y')
    plt.plot(time, q_z, label = 'q_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Attitude Controller Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()


# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"
    np.savetxt("../data/"+file_name, data, delimiter=",")

############################################################################################################
#-----------------------------------------TEST POSITION CONTROLLER-----------------------------------------#
# Set target state xyz. Run position controller to see how well acceleration error converges
############################################################################################################
tp = 0.
tfp = 4.
dt = 1/500

#Kp = np.diag([50.5, 52, 108.5]) 
#Kd = np.diag([65.0, 55.0, 63.0])
Kp = np.diag([10.9, 10.7, 10.77])
Kd = np.diag([4.5, 4.5,  5.53])

a_d = np.array([0., 0., 0.])

statep = np.zeros(13)
target_statep = np.zeros(13)

statep[0] = 0.
statep[1] = 0.
statep[2] = 0.

target_statep[0] = 0.1
target_statep[1] = 0.1
target_statep[2] = 0.1

error_a = np.empty((0, 2))
ae_x = np.empty((0, 2))
ae_y = np.empty((0, 2))
ae_z = np.empty((0, 2))
runningP = True
testingP = runningP

while runningP:
    a_e, a = pos.getAccelError(statep, target_statep, a_d, Kp, Kd)
    # print(a_e)
    ae_x = np.vstack((ae_x, np.array([tp, a_e[0]])))
    ae_y = np.vstack((ae_y, np.array([tp, a_e[1]])))
    ae_z = np.vstack((ae_z, np.array([tp, a_e[2]])))
    accel_error = np.array([tp, np.linalg.norm(a_e)])
    error_a = np.vstack((error_a, accel_error))
    statep[0:3] = statep[0:3] + statep[3:6] * dt
    statep[3:6] = statep[3:6] + a * dt

    # Update time
    tp += dt 

    # If time exceeds final time then stop simulator
    if tp >= tfp:
        runningP = False

if testingP == True:
    # plot the error to see how well the gains converge
    time = error_a[:,0]
    a_e = error_a[:, 1]
    a_x = ae_x[:, 1]
    a_y = ae_y[:, 1]
    a_z = ae_z[:, 1]
    
    plt.figure(2)
    plt.plot(time, a_e, label = 'a_e magnitude')
    plt.plot(time, a_x, label = 'a_x')
    plt.plot(time, a_y, label = 'a_y')
    plt.plot(time, a_z, label = 'a_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Position Controller Convergence')
    plt.axis([0, tfp, -2, 2])

    plt.legend()
    plt.grid(True)
    plt.show()

