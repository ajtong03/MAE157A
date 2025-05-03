import numpy as np
import random
from dynamics import dynamics   # import the class, not the module
import quaternionfunc as qf
# import controller as ctrl
import constants

# make a local dynamics instance (match your main’s params & dt!)
_dyn = dynamics(params=[9.81], dt=1.0/50.0)

#------------------------------ POSITION CONTROLLER---------------------------#
#-----takes the trajectory and outputs the corresponding attitude values-----
def posController1(traj):
    traj_length = len(traj)

    # just test 30 points
    num_points = traj_length
    inc = round(traj_length / num_points)
    attitude = [[0 for _ in range(3)] for _ in range(num_points)]

    for i in range(0, traj_length, inc):
        j = i * inc
        pos = traj[j][0:3]
        vel = traj[j][3:6]
        accel = traj[j][6:9]
        roll, pitch, yaw = compute_euler_angles(pos, vel, accel)
        attitude[i] = [roll, pitch, yaw]
    return attitude

#------------------------------ POSITION CONTROLLER---------------------------#
#-----takes the trajectory and ouutputs the corresponding attitude values and thrust-----
def posController2(state, p_d, v_d):
    Kp = constants.Kp_p
    Kd = constants.Kd_p

    p = state[0:3]
    v = state[3:6]
    # q = state[6:10]

    p_e = p_d - p
    v_e = v_d - v

    # compute force
    # ACCEL TERM ??
    F_des = -np.matmul(Kp, p_e) - np.matmul(Kd, v_e) + np.arr([0, 0, _dyn.g])

    # check that this doesn't exceed max thrust
    if np.linalg.norm(F_des) > constants.max_thrust:
        F_des = F_des * abs(constants.max_thrust /F_des)
    
    # NEED A CASE FOR IF F_des < 0?

    
    # compute desired quaternion using orthogonality
    z_hat = qf.unitQuat(F_des)
    x_hat =qf.unitQuat(np.cross(z_hat, np.cross(np.array([1,0,0]), z_hat)))
    y_hat = qf.unitQuat(np.cross(z_hat, x_hat))

    #rotation matrix
    R = np.column_stack(x_hat, y_hat, z_hat)
    q_d = qf.unitQuat(qf.R_to_quat(R))
    
    thrust = np.linalg.norm(F_des)

    return q_d, thrust

#---------------ATTITUDE CONTROLLER------------------------
def attController(state, q_d, w_d):
    Kp = constants.Kp_a
    Kd = constants.Kd_a

    q = state[6:10]
    torque = computeTorqueNaive(Kp, Kd, q, q_d, w_d)

    return torque


#------------------------------ GAIN FINDER ---------------------------#
#given desired attitude values, the optimal gains are found. these gains will be constant for all trajectories
def setAttController(state, attitude, thrust):
    # n is the number of gain combos to try
    n = 100
    max_errors =  np.zeros(n)
    gains = np.zeros(n)

    Kp_range = (1, 10)
    Kd_range = (1, 10)


    for i in range(n):
        stateNew = state
        Kp = np.diag([random.uniform(*Kp_range) for i in range(3)])
        Kd = np.diag([random.uniform(*Kd_range) for i in range(3)])
        gains[i] = [Kp, Kd]

        error = np.zeros(attitude.len)
        for j in attitude:
            roll, pitch, yaw = attitude[j]
            
            testState = stateNew
            q_d = qf.euler_to_quat(roll, pitch, yaw)
            # get actual from prop fxn in dynamics
            torque = computeTorqueNaive(Kp, Kd, testState[6:10], q_d, testState[10:13], 0) 

            # Obtain motor forces so new state can be propogated
            f = getForces(gains[i], torque, thrust)
            stateNew = _dyn.propagate(testState, f)
            q_a = stateNew[10:13] 

            temp = qf.error(q_a, q_d)

            # store the magnitude of the error
            error[j] = np.linalg.norm(temp)
        
        max_errors[i] = max(error)
    
    index_min = max_errors.index(min(max_errors))
    Kp_opt = gains[index_min][0]
    Kd_opt = gains[index_min][1]

    return Kp_opt, Kd_opt

#----------COMPUTE MOTOR FORCES GIVEN CONTROL GAINS AND RESULTANT TORQUES----------
def getForces(gains, torque, thrust):
    kp, kd = gains
    alloc_matrix = setAllocMat(kd)
    
    # inverse of allocation matrix
    alloc_matrix_inv = np.linalg.inv(alloc_matrix)

    forces = np.concatenate(thrust, torque)

    # motor forces
    f = np.matmul(alloc_matrix_inv, forces)

    # make sure motor forces are within the allowed thrusts
    f = np.clip(f, min=constants.min_thrust, max = constants.max_thrust)
    return f


#--------------COMPUTE TORQUES----------------
def computeTorqueNaive(Kp, Kd, q_act, q_d, w, w_d):
    q_e = qf.error(q_act, q_d)
    w_e = w - w_d

    # real part
    alpha = q_e[0]
    # vector part
    vec = q_e[1:]

    # DO THE ERRORS NEED TO BE TRANSPOSED
    torque = -alpha * np.matmul(Kp, vec) - np.matmul(Kd, w_e)
    return torque

#--------COMPUTE EULER ANGLES FROM POS, VEL, ACCEL-----------
# given position, velocity, and acceleration, find the corresponding roll, pitch, yaw

def compute_euler_angles(pos, vel, accel, gravity=np.array([0, 0, -9.81])):
    # Normalize forward direction (velocity)
    forward = vel / np.linalg.norm(vel)

    # Remove gravity from acceleration to estimate "up" direction
    up_est = accel - gravity
    up = up_est / np.linalg.norm(up_est)

    # Right direction = up × forward
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    # Recompute up to ensure orthogonality
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    # Construct rotation matrix: columns are right, up, forward
    R = np.column_stack((right, up, forward))

    # convert rotation matrix to euler angles
    phi = np.atan2(R(2,3), R(1,3)) 
    theta = np.asin(-R(3,3)) 
    psi = np.atan2(R(3,2), R(3,1)) 

    return phi, theta, psi # Optional: return in any order you prefer

def setAllocMat(kd):
    # allocation matrix
    r = _dyn.l
    alloc_mat = np.array([[1,1,1,1], 
                    [r, -r, r, -r], 
                    [-r, -r, r, r], 
                    [kd * r, -kd * r, -kd * r, kd * r]])
    
    return alloc_mat
