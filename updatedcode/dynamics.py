import numpy as np
from math import sin, cos, sqrt, tan

class quadrotor:
    def __init__(self, Ts=1.0/50.0, USE_PWM=0, USE_PID=0):
        # Simulation timestep
        self.Ts = Ts
        # Physical constants
        self.g = 9.81
        self.m = 1.4
        self.l = 0.56
        self.kd = 0.0000013858  # drag coeff
        self.kt = 0.000013328   # thrust coeff
        self.jp = 0.044         # prop inertia
        self.jx = 0.05
        self.jy = 0.05
        self.jz = 0.24
        # Control flags
        self.USE_PWM = USE_PWM
        self.USE_PID = USE_PID
        # Desired setpoints
        self.x_des = self.y_des = self.z_des = 0.0
        self.phi_des = self.theta_des = self.psi_des = 0.0
        # Integral terms
        self.x_error_sum = self.y_error_sum = self.z_error_sum = 0.0
        self.phi_error_sum = self.theta_error_sum = self.psi_error_sum = 0.0
        # State vector: [phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z]
        self.state = [0.0]*12
        self.input_vector = [0.0]*4
        self.time_elapse = 0.0

    def des_xyz(self, x=0.0, y=0.0, z=0.0):
        self.x_des, self.y_des, self.z_des = x, y, z

    def rotateGFtoBF(self, PHI, THETA, PSI, X, Y, Z):
        X_ = cos(PSI)*cos(THETA)*X + sin(PSI)*cos(THETA)*Y - sin(THETA)*Z
        Y_ = (cos(PSI)*sin(PHI)*sin(THETA) - cos(PHI)*sin(PSI))*X + \
              (sin(PHI)*sin(PSI)*sin(THETA) + cos(PHI)*cos(PSI))*Y + (cos(THETA)*sin(PHI))*Z
        Z_ = (cos(PHI)*cos(PSI)*sin(THETA) + sin(PHI)*sin(PSI))*X + \
              (cos(PHI)*sin(PSI)*sin(THETA) - cos(PSI)*sin(PHI))*Y + (cos(PHI)*cos(THETA))*Z
        return X_, Y_, Z_

    def PID_position(self):
        # X, Y set
        x_bf, y_bf, z_bf = self.rotateGFtoBF(self.state[0], self.state[1], self.state[2],
                                             self.state[9], self.state[10], self.state[11])
        x_err = self.x_des - x_bf
        self.x_error_sum += x_err
        theta_cmd = - (self.x_error_sum * 0.25 + x_err * 0.35 - 0.35 * self.state[6])
        self.theta_des = max(min(theta_cmd, 0.785), -0.785)
        # Y
        y_err = self.y_des - y_bf
        self.y_error_sum += y_err
        phi_cmd = self.y_error_sum * 0.25 + y_err * 0.35 - 0.35 * self.state[7]
        self.phi_des = max(min(phi_cmd, 0.785), -0.785)
        # Z
        z_err = self.z_des - z_bf
        self.z_error_sum += z_err
        u1 = (self.m*self.g + 5.88*z_err - 5.05*self.state[8]) / (cos(self.state[1])*cos(self.state[0]))
        self.input_vector[0] = max(min(u1, 43.5), 0.0)

    def PID_attitude(self):
        # Roll (phi)
        phi_err = self.phi_des - self.state[0]
        self.phi_error_sum += phi_err
        p_cmd = 4.5*phi_err + 0.0*self.phi_error_sum + 0.0*self.state[3]
        self.p_des = max(min(p_cmd, 0.8727), -0.8727)
        # Pitch (theta)
        theta_err = self.theta_des - self.state[1]
        self.theta_error_sum += theta_err
        q_cmd = 4.5*theta_err + 0.0*self.theta_error_sum + 0.0*self.state[4]
        self.q_des = max(min(q_cmd, 0.8727), -0.8727)
        # Yaw (psi)
        psi_err = self.psi_des - self.state[2]
        self.psi_error_sum += psi_err
        r_cmd = 4.5*psi_err + 0.0*self.psi_error_sum + 0.0*self.state[5]
        self.r_des = max(min(r_cmd, 10.0), -10.0)

    def PID_rate(self):
        # p-rate
        p_err = self.p_des - self.state[3]
        self.p_error_sum = getattr(self, 'p_error_sum', 0.0) + p_err
        u2 = 2.7*p_err + 1.0*self.p_error_sum - 0.01*self.state[10]
        self.input_vector[1] = max(min(u2, 6.25), -6.25)
        # q-rate
        q_err = self.q_des - self.state[4]
        self.q_error_sum = getattr(self, 'q_error_sum', 0.0) + q_err
        u3 = 2.7*q_err + 1.0*self.q_error_sum - 0.01*self.state[11]
        self.input_vector[2] = max(min(u3, 6.25), -6.25)
        # r-rate
        r_err = self.r_des - self.state[5]
        self.r_error_sum = getattr(self, 'r_error_sum', 0.0) + r_err
        u4 = 2.7*r_err + 1.0*self.r_error_sum - 0.01*self.state[8]
        self.input_vector[3] = max(min(u4, 2.25), -2.25)

    def quad_motor_speed(self):
        w1 = self.input_vector[0]/(4*self.kt) + self.input_vector[2]/(2*self.kt*self.l) + self.input_vector[3]/(4*self.kd)
        w2 = self.input_vector[0]/(4*self.kt) - self.input_vector[1]/(2*self.kt*self.l) - self.input_vector[3]/(4*self.kd)
        w3 = self.input_vector[0]/(4*self.kt) - self.input_vector[2]/(2*self.kt*self.l) + self.input_vector[3]/(4*self.kd)
        w4 = self.input_vector[0]/(4*self.kt) + self.input_vector[1]/(2*self.kt*self.l) - self.input_vector[3]/(4*self.kd)
        # Thrust & moments
        self.T = self.kt*(w1+w2+w3+w4)
        self.Mx = self.kt*self.l*(w4-w2)
        self.My = self.kt*self.l*(w1-w3)
        self.Mz = self.kd*(w1+w3-w2-w4)

    def step(self, state, input_vec):
        # Integrate one step using thrust/moments
        self.state = state
        self.quad_motor_speed()
        phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z = self.state
        # Translational acc
        acc = np.array([
            (-sin(theta)*self.T)/self.m,
            (cos(theta)*sin(phi)*self.T)/self.m,
            (cos(phi)*cos(theta)*self.T)/self.m - self.g
        ])
        x_dot += acc[0]*self.Ts
        y_dot += acc[1]*self.Ts
        z_dot += acc[2]*self.Ts
        x += x_dot*self.Ts
        y += y_dot*self.Ts
        z = max(0.0, z + z_dot*self.Ts)
        # Angular acc
        omega = np.array([p, q, r])
        domega = np.linalg.inv(np.diag([self.jx,self.jy,self.jz])) @ (
            np.array([self.Mx,self.My,self.Mz]) - np.cross(omega, np.diag([self.jx,self.jy,self.jz])@omega)
        )
        p += domega[0]*self.Ts
        q += domega[1]*self.Ts
        r += domega[2]*self.Ts
        # Euler integration
        phi += (p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r)*self.Ts
        theta += (cos(phi)*q - sin(phi)*r)*self.Ts
        psi += ((sin(phi)/cos(theta))*q + (cos(phi)/cos(theta))*r)*self.Ts
        self.state = [phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z]
        self.time_elapse += self.Ts
        return self.state, self.T


    def time_elapsed(self):
        return self.time_elapse