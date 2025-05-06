import constants
import numpy as np
import quaternionfunc as qf
from dynamics import dynamics as dyn 

class AttitudeController:
    def __init__(self, params, dt):
        # Simulation parameters
        self.g = params[0]            # gravity (m/s^2)
        self.m = 0.847                # Mass (kg)
        # Inertia Tensor (3×3) (kg·m^2)
        self.J = np.array([
            [ 1.89e-3, -5.78e-6,  1.07e-7],
            [-5.78e-6,  2.496e-3, -4.01e-4],
            [-1.07e-7, -2.01e-7,  4.55e-3]
        ])
        self.l = 0.07                 # Moment Arm (m)
        self.c = 0.0131               # Propeller Drag Coefficient (N·m/(N)^2)
        self.dt = dt                  # integration timestep (s)

        self.minThrust = 0.05433327 * 9.81 #N
        self.maxThrust = 0.392966325 * 9.81 #N



    #-------------------------ATTITUDE CONTROLLER----------------------------------#
    def attController(self, state, target_state, Kp, Kd):
        #Kp = constants.Kp_a
        #Kd = constants.Kd_a

        q = state[6:10]
        w = state[10:13]
        q_d = target_state[6:10]
        w_d = target_state[10:13]
        #torque = self.computeTorqueNaive(Kp, Kd, q, q_d, w, w_d)
        torque = self.computeTorqueNaive(Kp, Kd, q, q_d, w, w_d)

        return torque


    #-------------------------------------------------- GAIN FINDER GIVEN FULL TRAJECTORY-----------------------------------------------#
    #given desired attitude values from position controller, the optimal gains are found. these gains will be constant for all trajectories
    def setAttController(self, state, attitude, thrust):
        # n is the number of gain combos to try
        n = 100
        max_errors =  np.zeros(n)
        gains = np.zeros((n, 2))

        Kp_range = (1, 25)
        Kd_range = (1, 25)


        for i in range(n):
            state_current = state
            Kp = np.diag([np.random.uniform(*Kp_range) for i in range(3)])
            Kd = np.diag([np.random.uniform(*Kd_range) for i in range(3)])
            kpval = Kp[0][0]
            kdval = Kd[0][0]
            gains[i] =[kpval, kdval]

            error = np.zeros(attitude.len)
            for j in attitude:
                q_d = attitude[j]
                
                testState = state_current
                torque = self.computeTorqueNaive(Kp, Kd, testState[6:10], q_d, testState[10:13], 0) 

                # Obtain motor forces so new state can be propogated
                f = self.getForces(Kp, Kd, torque, thrust)
                state_current = dyn.propagate(testState, f, self.dt)
                print('state')
                print(testState[6:10])
                print('pause')
                q_a = state_current[6:10] 
                print(q_a)

                temp = qf.error(q_a, q_d)

                # store the magnitude of the error
                error[j] = np.linalg.norm(temp)
            
            max_errors[i] = max(error)
        
        index_min = max_errors.index(min(max_errors))
        Kp_opt = gains[index_min][0]
        Kd_opt = gains[index_min][1]

        return Kp_opt, Kd_opt

#-------------------------------------------------- GAIN FINDER GIVEN SINGLE ATTITUDE GOAL-----------------------------------------------#
    #given desired attitude value,  optimal gains are found. these gains will be constant for all trajectories 
    def setAttController2(self, state, attitude):
        dynam = dyn(np.array([9.81]), self.dt)
        # n is the number of gain combos to try
        n = 500
        max_errors =  np.zeros(n)
        gains = np.zeros((n, 2))

        Kp_range = (1, 50)
        Kd_range = (1, 50)

        max_errors = np.zeros(n)
        
        t = 0
        tf = 5
        for i in range(n):
            Kp = np.diag([np.random.uniform(*Kp_range) for i in range(3)])
            Kd = np.diag([np.random.uniform(*Kd_range) for i in range(3)])
            kpval = Kp[0][0]
            kdval = Kd[0][0]
            gains[i] =[kpval, kdval]

            q_d = attitude
            err_min = np.linalg.norm(qf.error(state[6:10], q_d))
            testState = state
            while t < tf:
                torque = self.computeTorqueNaive(Kp, Kd, testState[6:10], q_d, testState[10:13], 0) 
                # Obtain motor forces so new state can be propogated
                f = self.getForces(torque)
                # print('forces')
                # print(f)
                print('states')
                print(testState[6:10])
                newState = dynam.propagate(testState, f, self.dt)
                q_a = newState[6:10] 
                print(q_a)
                err = qf.error(q_a, q_d)
                print('error')
                # store the magnitude of the error
                err = np.linalg.norm(err)
                print(err)
                if err < err_min:
                    err_min = err
                t += self.dt
                testState = newState
            max_errors[i] = err_min
        
        index_min = np.argmin(max_errors)
        Kp_opt = gains[index_min][0]
        Kd_opt = gains[index_min][1]

        #print(max_errors)
        return Kp_opt, Kd_opt

    #----------COMPUTE MOTOR FORCES GIVEN CONTROL GAINS AND RESULTANT TORQUES----------#
    def getForces(self, torque):
        # self.alloc_matrix = self.setAllocMat(kd)
        r = self.l
        self.alloc_matrix = np.array([[r, r, -r, -r], 
                                       [-r, r, r, -r], 
                                       [self.c,-self.c, self.c,-self.c]]) 

        # inverse of allocation matrix
        f = np.linalg.lstsq(self.alloc_matrix, torque)
        f = f[0]
        # make sure motor forces are within the allowed thrusts
        f = np.clip(f, min=constants.min_thrust, max = constants.max_thrust)
        return f


    #------------------------COMPUTE TORQUES--------------------------#
    @staticmethod
    def computeTorqueNaive(Kp, Kd, q_act, q_d, w, w_d):
        q_e = qf.error(q_act, q_d)
        w_e = w - w_d

        # real part
        alpha = q_e[0]
        # vector part
        vec = q_e[1:]

        torque = -alpha * np.matmul(Kp, vec) - np.matmul(Kd, w_e)
        return np.array(torque)

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

    def setAllocMat(self, Kd):
        # allocation matrix
        r = 0.07
        kd = Kd[0][0]
        alloc_mat = np.array([[1,1,1,1],
                            [r, -r, r, -r], 
                            [-r, -r, r, r], 
                            [kd * r, -kd * r, -kd * r, kd * r]])
        
        
        return alloc_mat