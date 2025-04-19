import numpy as np



class dynamics:
    def __init__(self, params, dt):
        # Simulation parameters
        self.g = params[0]            # gravity (m/s^2)
        self.m = 1.0                  # Mass (kg)
        self.J = np.diag([0.02, 0.02, 0.04])  # Inertia Tensor (kg·m^2)
        self.l = 0.1                  # Moment Arm (m)
        self.c = 0.01                 # Propeller Drag Coefficient (N·m/(N)^2)
        self.dt = dt                 # integration timestep (s)

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
        # State: [x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz]
        q = state[6:10]
        R = self.quat_to_rot(q)

        # Translational dynamics
        T = np.sum(f)
        acc = (R @ np.array([0, 0, T])) / self.m - np.array([0, 0, self.g])

        # Rotational dynamics
        # Thrust-based moments around x and y axes
        Mx = self.l * (f[1] - f[3])
        My = self.l * (f[2] - f[0])
        # Drag-based yaw moment
        Mz = self.c * (f[0] - f[1] + f[2] - f[3])
        M = np.array([Mx, My, Mz])

        # Angular acceleration: inv(J) * M
        omega = state[10:13]
        domega = np.linalg.inv(self.J) @ (M - np.cross(omega, self.J @ omega))

        # Quaternion kinematics: 0.5 * Omega(omega) * q
        qw, qx, qy, qz = q
        Omega = np.array([
            [0,   -omega[0], -omega[1], -omega[2]],
            [omega[0], 0,    omega[2], -omega[1]],
            [omega[1], -omega[2], 0,    omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        dq = 0.5 * Omega @ q

        rates = np.zeros(13)
        rates[0:3] = state[3:6]         # vel
        rates[3:6] = acc               # acc
        rates[6:10] = dq               # quaternion rate
        rates[10:13] = domega          # angular accel
        return rates

    def propagate(self, state, f, dt=None):
        # Allow optional override of timestep
        step = dt if dt is not None else self.dt
        return state + step * self.rates(state, f)


    def quat_to_rot(self, q):
        w, i, j, k = q

        R_x =[(w**2 + i**2 - j**2 - k**2), (2 * (i*j + w*k)), (2 * (i*k - w*j))]
        R_y = [(2 * (i*j - w*k)), (w**2 - i**2 + j**2 - k**2), (2 * (j*k + w*i))]
        R_z = [(2 * (i*k + w*j)), (2 * (j*k - w*i)), (w**2 - i**2 - j**2 -+k**2)] 

        return np.array([R_x, R_y, R_z])