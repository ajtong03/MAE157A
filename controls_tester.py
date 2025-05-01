import numpy as np
import random
from dynamics import dynamics   # import the class, not the module
import quaternionfunc as qf

# make a local dynamics instance (match your main’s params & dt!)
_dyn = dynamics(params=[9.81], dt=1.0/50.0)
A_mat = np.array([[1,1,1,1], [0, _dyn.l,0,-_dyn.l], [-_dyn.l,0, _dyn.l,0], [_dyn.c,-_dyn.c, _dyn.c,-_dyn.c]])

def posController(traj):
    traj_length = len(traj)

    # just test 30 points
    num_points = 30
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

def attController(attitude):
    # n is the number of gain combos to try
    n = 100
    max_errors =  np.zeros(n)
    gains = np.zeros(n)

    Kp_range = (1, 10)
    Kd_range = (1, 10)

    for i in range(n):
        Kp = np.diag([random.uniform(*Kp_range) for i in range(3)])
        Kd = np.diag([random.uniform(*Kd_range) for i in range(3)])
        gains[i] = [Kp, Kd]

        error = np.zeros(attitude.len)
        for j in attitude:
            roll, pitch, yaw = attitude[j]

            q_d = qf.euler_to_quat(roll, pitch, yaw)
            # get actual from prop fxn in dynamics
            #REPLACE THIS
            q_a = 1

            temp = qf.error(q_a, q_d)
            # store the magnitude of the error
            error[j] = np.linalg.norm(temp)
        
        max_errors[i] = max(error)
    
    index_min = max_errors.index(min(max_errors))
    Kp_opt = gains[index_min][0]
    Kd_opt = gains[index_min][1]

    return Kp_opt, Kd_opt


        
        

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

