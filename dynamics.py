import numpy as np

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
        self.c = 0.0131               # Propeller Drag Coefficient (N·m/(N)^2)
        self.dt = dt                  # integration timestep (s)

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
        R = self.quat_to_rot(q)

        # Translational dynamics
        T = np.sum(f)
        acc = (R @ np.array([0, 0, T])) / self.m - np.array([0, 0, self.g])

        # Rotational dynamics
        Mx = self.l * (f[1] - f[3])
        My = self.l * (f[2] - f[0])
        Mz = self.c * (f[0] - f[1] + f[2] - f[3])
        M = np.array([Mx, My, Mz])

        omega = state[10:13]
        domega = np.linalg.inv(self.J) @ (M - np.cross(omega, self.J @ omega))

        # Quaternion kinematics
        qw, qx, qy, qz = q
        Omega = np.array([
            [0,      -omega[0], -omega[1], -omega[2]],
            [omega[0],  0,       omega[2], -omega[1]],
            [omega[1], -omega[2], 0,        omega[0]],
            [omega[2],  omega[1], -omega[0], 0]
        ])
        dq = 0.5 * Omega @ q

        rates = np.zeros(13)
        rates[0:3]   = state[3:6]    # velocity
        rates[3:6]   = acc          # acceleration
        rates[6:10]  = dq           # quaternion rate
        rates[10:13] = domega       # angular acceleration
        return rates

    def propagate(self, state, f, dt=None):
        step = dt if dt is not None else self.dt
        return state + step * self.rates(state, f)

    @staticmethod
    def quat_to_rot(q):
        w, i, j, k = q
        R_x = [w*w + i*i - j*j - k*k, 2*(i*j + w*k),       2*(i*k - w*j)]
        R_y = [2*(i*j - w*k),         w*w - i*i + j*j - k*k, 2*(j*k + w*i)]
        R_z = [2*(i*k + w*j),         2*(j*k - w*i),         w*w - i*i - j*j + k*k]
        return np.array([R_x, R_y, R_z])