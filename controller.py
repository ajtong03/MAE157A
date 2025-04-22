import quaternionfunc
import numpy as np
import random

def quat_angle_err(q_act, q_d):
# compute angle of error
    q_err = quaternionfunc.error(q_act, q_d)
    # normalize q_err so it's in terms of unit quat
    q_err /= np.linalg.norm(q_err)
    # angle in radians
    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0)) 

    return angle

#compute overshoot and settling time for tuning
# set the time limit to 10s and the overshoot threshold to 5%
def overshoot_settime(q_act, q_d, time = 10, thresh = 0.05):
    # compute angle error in i, j, k
    angle_err = [quat_angle_err(q, q_d) for q in q_act]
    angle_err = np.array(angle_err)
    
    # compute overshoot
    overshoot = np.max(angle_err)

    set_time = time[-1]

    for i in range(len(angle_err)):
        if np.all(angle_err[i:] < thresh):
            set_time = time[i]
            break
    return overshoot, set_time
 

def computeGains(q_act, q_d):
    # starting gain ranges
    Kp_range = (1, 20)  # roll, pitch, yaw
    Kd_range = (1, 20)
    # REASONABLE RANGE FOR LAMBDA=?
    lambda_range = (0, 1)

    Kp_opt = np.diag([1, 1, 1])
    Kd_opt = np.diag([1, 1, 1])
    lambda_opt = np.diag([1, 1, 1])

    #set variable to track optimal performance value
    perf_opt = float('inf')
    over_opt = float('inf')
    set_opt = float('inf') 

    #run 1000 iterations to tune Kp and Kd
    for i in range(1000):

        #set random parameters for Kp, Kd (roll, pitch, yaw), and lambda 
        Kp = np.diag([random.uniform(*Kp_range) for i in range(3)])
        Kd = np.diag([random.uniform(*Kd_range) for i in range(3)])
        lambda_mat = np.diag([random.uniform(*lambda_range)] for i in range(3))

        # q_e = quaternionfunc.error(q_act, q_d)

        #performance parameters
        overshoot, set_time = overshoot_settime(q_act, q_d)
        # EDIT THIS TO GIVE DIFFERENT WEIGHTS TO EACH PERFORMANCE PARAM
        perf = overshoot + set_time

        if(perf < perf_opt):
            perf_opt = perf
            Kp_opt = Kp
            Kd_opt = Kd
            over_opt = overshoot
            set_opt = set_time
        
            # narrow the range if gains improve performance
            Kp_range = (Kp_opt[0, 0] - 0.5, Kp_opt[0, 0] + 0.5)
            Kd_range = (Kd_opt[0, 0] - 0.5, Kd_opt[0, 0] + 0.5)
        
        #Increase the range if no improvement
        elif i % 100 == 0 and perf_opt == float('inf'):
            Kp_range = (0.8*Kp_range(0), 1.2*Kp_range(1))
            Kd_range = (0.8*Kd_range(0), 1.2*Kd_range(1))
    
    return Kp_opt, Kd_opt, over_opt, set_opt, lambda_opt

#IMPLEMENT THIS LATER
def computeLambda(range, lambda_opt):
    lambda_mat = np.diag([1, 1, 1])
    return lambda_mat


def computeTorqueNaive(Kp, Kd, q_act, q_d, w, w_d):
    q_e = quaternionfunc.error(q_act,q_d)
    w_e = w - w_d
    torque = -q_e(0) * Kp * q_e(1) - Kd * w_e
    return torque

def computeTorque(Kp, Kd, lambda_opt, q_act, q_d, w, w_d):
    # REPLACE WITH ACTUAL VALUE OF Re
    Re = 1

    q_e = quaternionfunc.error(q_act,q_d)
    w_e = w - Re * w_d
    q_deriv = quaternionfunc.deriv(q_act, w)

    sgn = 1
    if(q_e <= 0):
        sgn = -1

    torque = -sgn * q_e(0) * Kp * q_e(1) - Kd * w_e - lambda_opt * sgn * q_e(0) * q_deriv(1)
    return torque
    