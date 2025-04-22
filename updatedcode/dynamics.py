import numpy as np
from math import sin, cos, sqrt, tan

class quadrotor:
    def __init__(self, Ts=1.0/50.0, USE_PWM=0, USE_PID=0):
        # Simulation timestep
        self.Ts = Ts
        # Physical constants
        self.g = 9.81; self.m = 1.4; self.l = 0.56
        self.kd = 1.3858e-6; self.kt = 1.3328e-5; self.jp = 0.044
        self.jx = self.jy = 0.05; self.jz = 0.24
        # Control flags
        self.USE_PWM = USE_PWM; self.USE_PID = USE_PID
        # Desired setpoints
        self.x_des = self.y_des = self.z_des = 0.0
        self.phi_des = self.theta_des = self.psi_des = 0.0
        # Integral terms
        self.x_error_sum = self.y_error_sum = self.z_error_sum = 0.0
        self.phi_error_sum = self.theta_error_sum = self.psi_error_sum = 0.0
        # State: [phi,theta,psi,p,q,r, x_dot,y_dot,z_dot, x,y,z]
        self.state = [0.0]*12
        self.input_vector = [0.0]*4
        self.time_elapse = 0.0

    def des_xyz(self, x=0.0, y=0.0, z=0.0):
        self.x_des, self.y_des, self.z_des = x, y, z

    def rotateGFtoBF(self, PHI, THETA, PSI, X, Y, Z):
        X_ = cos(PSI)*cos(THETA)*X + sin(PSI)*cos(THETA)*Y - sin(THETA)*Z
        Y_ = (cos(PSI)*sin(PHI)*sin(THETA)-cos(PHI)*sin(PSI))*X + \
              (sin(PHI)*sin(PSI)*sin(THETA)+cos(PHI)*cos(PSI))*Y + (cos(THETA)*sin(PHI))*Z
        Z_ = (cos(PHI)*cos(PSI)*sin(THETA)+sin(PHI)*sin(PSI))*X + \
              (cos(PHI)*sin(PSI)*sin(THETA)-cos(PSI)*sin(PHI))*Y + (cos(PHI)*cos(THETA))*Z
        return X_, Y_, Z_

    def PID_position(self):
        x_bf, y_bf, z_bf = self.rotateGFtoBF(*self.state[:3], *self.state[9:12])
        x_err = self.x_des - x_bf; y_err = self.y_des - y_bf; z_err = self.z_des - z_bf
        self.x_error_sum += x_err; self.y_error_sum += y_err; self.z_error_sum += z_err
        theta_cmd = -(0.35*x_err + 0.25*self.x_error_sum - 0.35*self.state[6])
        phi_cmd   =  (0.35*y_err + 0.25*self.y_error_sum - 0.35*self.state[7])
        u1        = (self.m*(self.g + 5.88*z_err - 5.05*self.state[8]))/ (cos(self.state[1])*cos(self.state[0]))
        self.theta_des = max(min(theta_cmd,  0.785), -0.785)
        self.phi_des   = max(min(phi_cmd,    0.785), -0.785)
        self.input_vector[0] = max(min(u1, 50.0), 0.0)

    def PID_attitude(self):
        phi_err = self.phi_des - self.state[0]; theta_err = self.theta_des - self.state[1]; psi_err = self.psi_des - self.state[2]
        self.p_des = max(min(4.5*phi_err, 0.8727), -0.8727)
        self.q_des = max(min(4.5*theta_err, 0.8727), -0.8727)
        self.r_des = max(min(2.0*psi_err, 10.0), -10.0)

    def PID_rate(self):
        self.input_vector[1] = max(min(2.7*(self.p_des - self.state[3]), 6.25), -6.25)
        self.input_vector[2] = max(min(2.7*(self.q_des - self.state[4]), 6.25), -6.25)
        self.input_vector[3] = max(min(2.7*(self.r_des - self.state[5]), 2.25), -2.25)

    def quad_motor_speed(self):
        u = self.input_vector
        w1 = u[0]/(4*self.kt) + u[2]/(2*self.kt*self.l) + u[3]/(4*self.kd)
        w2 = u[0]/(4*self.kt) - u[1]/(2*self.kt*self.l) - u[3]/(4*self.kd)
        w3 = u[0]/(4*self.kt) - u[2]/(2*self.kt*self.l) + u[3]/(4*self.kd)
        w4 = u[0]/(4*self.kt) + u[1]/(2*self.kt*self.l) - u[3]/(4*self.kd)
        self.T  = self.kt*(w1+w2+w3+w4)
        self.Mx = self.kt*self.l*(w4-w2); self.My = self.kt*self.l*(w1-w3); self.Mz = self.kd*(w1+w3-w2-w4)

    def step(self, state, _):
        self.state = state; self.quad_motor_speed()
        phi,theta,psi,p,q,r,xd,yd,zd,x,y,z = self.state
        acc = np.array([-sin(theta)*self.T/self.m, cos(theta)*sin(phi)*self.T/self.m, cos(phi)*cos(theta)*self.T/self.m - self.g])
        xd+=acc[0]*self.Ts; yd+=acc[1]*self.Ts; zd+=acc[2]*self.Ts
        x+=xd*self.Ts; y+=yd*self.Ts; z=max(0,z+zd*self.Ts)
        omega=np.array([p,q,r]); I=np.diag([self.jx,self.jy,self.jz])
        domega=np.linalg.inv(I)@(np.array([self.Mx,self.My,self.Mz]) - np.cross(omega,I@omega))
        p+=domega[0]*self.Ts; q+=domega[1]*self.Ts; r+=domega[2]*self.Ts
        phi+=(p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r)*self.Ts
        theta+=(cos(phi)*q - sin(phi)*r)*self.Ts
        psi+=((sin(phi)/cos(theta))*q + (cos(phi)/cos(theta))*r)*self.Ts
        self.state=[phi,theta,psi,p,q,r,xd,yd,zd,x,y,z]
        self.time_elapse+=self.Ts
        return self.state,self.T

    def time_elapsed(self): return self.time_elapse