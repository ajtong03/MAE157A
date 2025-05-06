import constants
import numpy as np
import quaternionfunc as qf
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

    def posController2(self, state, target_state, a_d):
        Kp = np.diag([2, 2, 2])
        Kd = np.diag([3, 3, 3])

        p = state[0:3]
        v = state[3:6]

        p_e = target_state[0:3] - p
        v_e = target_state[3:6] - v

        # compute acceleration and desired thrust
        a = a_d - np.matmul(Kp, p_e) - np.matmul(Kd, v_e) + np.arr([0, 0, self.g])
        thrust = self.m * a

        # check that this doesn't exceed max thrust
        if np.linalg.norm(thrust) > constants.max_thrust:
            thrust = thrust * abs(constants.max_thrust /thrust)
        
        # compute desired orientation
        # a_hat is the acceleration unit vector
        # e represents the z-coordinate axis
        # e_T is the transpose of e
        a_hat = a / np.linalg.norm(a)
        e = np.array([0, 0, 1])
        e_T = e.reshape(-1, 1)
        w_d = np.arr([target_state[11], -target_state[10], 0])
        w_d = w_d.reshape(-1, 1)

        q_d = (1 / np.sqrt(2 * (1 + e_T @ a_hat))) * (np.array(1 + e_T @ a_hat, np.cross(e, a_hat)) @ w_d)

        return q_d, thrust