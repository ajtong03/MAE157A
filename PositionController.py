import constants
import numpy as np
from quaternionfunc import *
class PositionController:
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

    def posController(self, state, target_state, a_d: np.ndarray, j_d: np.ndarray):
        Kp = np.diag([22.5, 22, 18.5])
        Kd = np.diag([35.0, 35.0, 33.0])

        p = state[0:3]
        v = state[3:6]

        p_e = p - target_state[0:3]
        v_e = v - target_state[3:6]

        # compute acceleration and desired thrust
        a = a_d - Kp @ p_e - Kd @ v_e + np.array([0, 0, self.g])
        # a_hat is the acceleration unit vector
        a_hat = a / np.linalg.norm(a)

        thrust = self.m * np.linalg.norm(a)

        
        thrust = np.clip(thrust, self.minThrust, self.maxThrust)
        
        
        # compute desired orientation
        # e represents the z-coordinate axis
        # e_T is the transpose of e
        e = np.array([0, 0, 1])
        w_d = np.array([target_state[11], -target_state[10], 0])
        w_d = w_d.T

        multiplier = 1 / np.sqrt(2 * (1 + np.dot(e.T, a_hat)))
        qw =  1 + np.dot(e.T, a_hat)
        vec = np.cross(e, a_hat)
        q_d = multiplier * np.array([qw, vec[0], vec[1], vec[2]])

        R_d = quat_to_rot(q_d)
        print(R_d.T)
        ahat_dot = 1 / np.linalg.norm(a) * (j_d - np.dot(a_hat, j_d) * a_hat)
        w = R_d.T @ ahat_dot
        w_d = np.zeros(3)
        w_d[0] = -w[1]
        w_d[1] = w[0]
        w_d[2] = 0
        return q_d, w_d, thrust
    
    def getAccelError(self, state, target_state, a_d:np.ndarray, Kp, Kd):

        p = state[0:3]
        v = state[3:6]

        p_e = p - target_state[0:3]
        v_e = v - target_state[3:6] 

        # compute acceleration and desired thrust
        a = a_d - Kp @ p_e - Kd @ v_e + np.array([0, 0, self.g])

        a_e = a - a_d
        
        return a_e, a 