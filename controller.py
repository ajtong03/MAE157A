import numpy as np
import random
from dynamics import dynamics   # import the class, not the module
import quaternionfunc

# make a local dynamics instance (match your main’s params & dt!)
_dyn = dynamics(params=[9.81], dt=1.0/50.0)

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

# naive approach to computing gains  
def naiveComputeGains(q_act, q_d, f):
    Kp_range = (1, 20)
    Kd_range = (1, 20)

    # current best error (angle or norm—whatever you were using)
    q_e_opt = quaternionfunc.error(q_act, q_d)

    Kp_opt = np.diag([0, 0, 0])
    Kd_opt = np.diag([0, 0, 0])

    for _ in range(1000):
        Kp = np.diag([random.uniform(*Kp_range) for _ in range(3)])
        Kd = np.diag([random.uniform(*Kd_range) for _ in range(3)])

        # build a full‐size state vector, sticking q_act into the quaternion slot
        state_test = np.zeros(13)
        state_test[6:10] = q_act

        # propagate that test state under your candidate thrusts `f`
        state_new = _dyn.propagate(state_test, f)

        # pull out the new quaternion
        q_new = state_new[6:10]
        
        # measure the new error
        qe_new = quaternionfunc.error(q_new, q_d)

        # if it’s better, keep these gains
        if np.linalg.norm(qe_new) < np.linalg.norm(q_e_opt):
            q_e_opt = qe_new
            Kp_opt = Kp
            Kd_opt = Kd

    return Kp_opt, Kd_opt

# the torque needs to be applied to the orientation and q_act needs to be refed into this   
def computeGains(q_act, q_d):
    # starting gain ranges
    Kp_range = (1, 20)  # roll, pitch, yaw
    Kd_range = (1, 20)
    # REASONABLE RANGE FOR LAMBDA=?
    lambda_range = (0, 1)

    # set these so that if q_act is already at q_d, stay in orientation
    Kp_opt = np.diag([0, 0, 0])
    Kd_opt = np.diag([0, 0, 0])
    lambda_opt = np.diag([0, 0, 0])

    #set variable to track optimal performance value
    #q_e = quaternionfunc.error(q_act, q_d)
    over_opt, set_opt = overshoot_settime(q_act, q_d)
    perf_opt = over_opt + set_opt

    #run 1000 iterations to tune Kp and Kd
    for i in range(1000):

        #set random parameters for Kp, Kd (roll, pitch, yaw), and lambda 
        Kp = np.diag([random.uniform(*Kp_range) for i in range(3)])
        Kd = np.diag([random.uniform(*Kd_range) for i in range(3)])
        lambda_mat = np.diag([random.uniform(*lambda_range)] for i in range(3))

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
            # Kp_range = (Kp_opt[0, 0] - 0.5, Kp_opt[0, 0] + 0.5)
            # Kd_range = (Kd_opt[0, 0] - 0.5, Kd_opt[0, 0] + 0.5)
        
        #Increase the range if no improvement
        # elif i % 100 == 0 and perf_opt == float('inf'):
        #     Kp_range = (0.8*Kp_range(0), 1.2*Kp_range(1))
        #     Kd_range = (0.8*Kd_range(0), 1.2*Kd_range(1))
    
    return Kp_opt, Kd_opt, over_opt, set_opt, lambda_opt

#IMPLEMENT THIS LATER
def computeLambda(range, lambda_opt):
    lambda_mat = np.diag([1, 1, 1])
    return lambda_mat


def computeTorqueNaive(Kp, Kd, q_act, q_d, w, w_d):
    q_e = quaternionfunc.error(q_act, q_d)
    w_e = w - w_d

    # real part
    alpha = q_e[0]
    # vector part
    eps = q_e[1:]

    # torque = – α·(Kp·eps)  –  (Kd·w_e)
    torque = -alpha * (Kp.dot(eps)) - (Kd.dot(w_e))
    return torque

def computeTorque(Kp, Kd, lambda_opt, q_act, q_d, w, w_d, Re):
    # Re should be computed in main using dynamics.quat_to_rot(q_e)

    qe_mag, qe_vec = quaternionfunc.error(q_act,q_d)
    w_e = w - Re * w_d
    q_deriv = quaternionfunc.deriv(q_act, w)

    sgn = 1
    if(qe_mag < 0):
        sgn = -1

    torque = -sgn * qe_mag * Kp * qe_vec - Kd * w_e - lambda_opt * sgn * qe_mag * q_deriv(1)
    return torque
    