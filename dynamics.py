import numpy as np
from quaternionfunc import *
class dynamics:
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
        self.c = 0.2               # Propeller Drag Coefficient (N·m/(N)^2)
        self.dt = dt                  # integration timestep (s)

        self.minThrust = 0.05433327 * 9.81 #N
        self.maxThrust = 0.392966325 * 9.81 #N

        self.A = np.array([[self.l, self.l, -self.l, -self.l], 
                            [-self.l, self.l, self.l, -self.l], 
                            [self.c,-self.c, self.c,-self.c]]) 


    def print_specs(self):
        print("=== Drone Specifications ===")
        print(f"Mass: {self.m} kg")
        print(f"Gravity: {self.g} m/s^2")
        print(f"Inertia Tensor:\n{self.J}")
        print(f"Moment Arm: {self.l} m")
        print(f"Propeller Drag Coefficient: {self.c} N·m/N^2")
        print(f"Timestep: {self.dt} s")
        print("============================")

    def rates(self, state, f):
        # State: [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
        q = state[6:10]
        R = quat_to_rot(q)

        # Translational dynamics
        T = np.sum(f)
        acc = (R @ np.array([0, 0, T])) / self.m - np.array([0, 0, self.g])

        # Rotational dynamics
        M = self.A @ f

        omega = state[10:13]
        domega = np.linalg.inv(self.J) @ (M + np.cross(-omega, self.J @ omega))

        omega_q = np.array([0, *omega])
        dq = 0.5 * product(q, omega_q)
        rates = np.zeros(13)
        rates[0:3]   = state[3:6]    # velocity
        rates[3:6]   = acc          # acceleration
        rates[6:10]  = dq           # quaternion rate
        rates[10:13] = domega       # angular acceleration

        return rates

    # Using RK4 formula to propagate
    def propagate(self, state, f, dt=None):
        step = dt if dt is not None else self.dt

        k1 = self.rates(state, f)
        k2 = self.rates(state + 0.5 * step * k1, f)
        k3 = self.rates(state + 0.5 * step * k2, f)
        k4 = self.rates(state + step * k3, f)

        state += (step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state[6:10] /= np.linalg.norm(state[6:10])
        return state
   